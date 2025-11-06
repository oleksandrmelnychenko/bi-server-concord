#!/usr/bin/env python3
"""
Train Survival Analysis Model for Repurchase Prediction

Predicts WHEN customers will reorder specific products using:
- Weibull Accelerated Failure Time (AFT) model
- Customer repurchase history
- Product-specific purchase cycles

Output:
- Survival model that predicts time-to-next-purchase
- Reorder probability scores (0-1)
- Proactive alerts for customers overdue for reorders

Business Impact: Win back 616 "Lost" customers (~$1.4M revenue potential)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from lifelines import WeibullAFTFitter
from lifelines.utils import median_survival_times
import pickle
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/ml_features")
MODEL_DIR = Path("models/survival_analysis")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DUCKDB_PATH = DATA_DIR / "concord_ml.duckdb"
MODEL_PATH = MODEL_DIR / "weibull_repurchase_model.pkl"
ALERTS_PATH = MODEL_DIR / "reorder_alerts.csv"


class RepurchaseSurvivalAnalysis:
    """Survival analysis for predicting customer repurchase timing"""

    def __init__(self):
        self.model = WeibullAFTFitter()
        self.model_trained = False

    def load_data(self):
        """Load repurchase data from DuckDB"""
        logger.info("\n" + "="*80)
        logger.info("LOADING REPURCHASE DATA")
        logger.info("="*80)

        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)

        # Load interactions with repurchase patterns
        query = """
        SELECT
            customer_id,
            product_id,
            num_purchases,
            total_spent,
            days_since_last_purchase,
            purchase_span_days,
            implicit_rating,
            repurchase_likelihood,
            is_repeat_customer,
            is_at_risk,
            revenue_potential
        FROM ml_features.interaction_matrix
        WHERE num_purchases >= 2  -- Need at least 2 purchases to calculate intervals
        """

        df = conn.execute(query).df()

        # Calculate average inter-purchase interval
        df['avg_purchase_interval'] = df['purchase_span_days'] / (df['num_purchases'] - 1)

        # Create censoring indicator
        # Event = 1 if customer repurchased (observed)
        # Event = 0 if still waiting (censored)
        df['event_observed'] = (df['days_since_last_purchase'] < df['avg_purchase_interval'] * 2).astype(int)

        # Duration = days since last purchase
        df['duration'] = df['days_since_last_purchase']

        # Load customer features for covariates
        customer_query = """
        SELECT
            customer_id,
            recency_score,
            frequency_score,
            monetary_score,
            total_orders,
            lifetime_value,
            customer_segment,
            customer_tier
        FROM ml_features.customer_features
        """

        customer_df = conn.execute(customer_query).df()

        # Merge
        df = df.merge(customer_df, on='customer_id', how='left')

        conn.close()

        logger.info(f"✓ Loaded {len(df):,} customer-product repurchase records")
        logger.info(f"  Repeat customers: {df['customer_id'].nunique():,}")
        logger.info(f"  Products: {df['product_id'].nunique():,}")
        logger.info(f"  Event observed (repurchased): {df['event_observed'].sum():,}")
        logger.info(f"  Censored (waiting): {(1 - df['event_observed']).sum():,}")
        logger.info(f"  Mean purchase interval: {df['avg_purchase_interval'].mean():.1f} days")

        return df

    def prepare_training_data(self, df):
        """Prepare data for Weibull AFT model"""
        logger.info("\n" + "="*80)
        logger.info("PREPARING TRAINING DATA")
        logger.info("="*80)

        # Select features for survival model
        # These are covariates that might affect repurchase timing
        feature_cols = [
            'num_purchases',  # How many times they bought this product
            'total_spent',  # How much they spent
            'implicit_rating',  # Their affinity for this product
            'recency_score',  # How recently they purchased anything
            'frequency_score',  # How often they purchase
            'monetary_score',  # How valuable they are
            'avg_purchase_interval'  # Historical interval
        ]

        # Create training dataframe
        train_df = df[feature_cols + ['duration', 'event_observed']].copy()

        # Handle missing values
        train_df = train_df.fillna(train_df.mean())

        # Log transform highly skewed features
        train_df['total_spent_log'] = np.log1p(train_df['total_spent'])
        train_df['avg_purchase_interval_log'] = np.log1p(train_df['avg_purchase_interval'])

        # Drop original skewed columns
        train_df = train_df.drop(['total_spent', 'avg_purchase_interval'], axis=1)

        logger.info(f"✓ Training data prepared:")
        logger.info(f"  Samples: {len(train_df):,}")
        logger.info(f"  Features: {len(train_df.columns) - 2}")  # Exclude duration and event_observed
        logger.info(f"  Event rate: {train_df['event_observed'].mean():.2%}")

        return train_df, df

    def train(self, train_df):
        """Train Weibull AFT model"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING WEIBULL AFT MODEL")
        logger.info("="*80)

        start_time = datetime.now()
        logger.info(f"Training started at {start_time}...")

        # Fit Weibull AFT model
        self.model.fit(
            train_df,
            duration_col='duration',
            event_col='event_observed',
            show_progress=True
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"✓ Training complete in {duration:.1f} seconds")

        self.model_trained = True

        # Print model summary
        logger.info("\nModel Parameters:")
        # params_ is a Series, use .item() to extract scalar
        params_dict = self.model.params_.to_dict()
        if 'lambda_' in params_dict:
            logger.info(f"  Lambda (scale): {params_dict['lambda_']:.4f}")
        if 'rho_' in params_dict:
            logger.info(f"  Rho (shape): {params_dict['rho_']:.4f}")

        # Print feature importance (coefficients)
        logger.info("\nFeature Coefficients (λ):")
        for param, value in params_dict.items():
            if param not in ['rho_', 'lambda_']:
                logger.info(f"  {param}: {value:.4f}")

        return self.model

    def evaluate(self, train_df):
        """Evaluate model performance"""
        logger.info("\n" + "="*80)
        logger.info("EVALUATING MODEL")
        logger.info("="*80)

        # Concordance index (C-index)
        # Measures how well the model ranks survival times
        # 1.0 = perfect, 0.5 = random
        c_index = self.model.concordance_index_
        logger.info(f"Concordance Index (C-index): {c_index:.4f}")

        if c_index > 0.75:
            logger.info("  ✅ Excellent! Model has strong predictive power")
        elif c_index > 0.65:
            logger.info("  ✓ Good! Model is useful for predictions")
        elif c_index > 0.55:
            logger.info("  ⚠️ Moderate. Model has some predictive power")
        else:
            logger.info("  ❌ Weak. Model needs improvement")

        # AIC (Akaike Information Criterion)
        # Lower is better
        aic = self.model.AIC_
        logger.info(f"AIC: {aic:.2f}")

        # Log-likelihood
        logger.info(f"Log-likelihood: {self.model.log_likelihood_:.2f}")

    def predict_repurchase_times(self, df, original_df):
        """Predict when customers will repurchase"""
        logger.info("\n" + "="*80)
        logger.info("PREDICTING REPURCHASE TIMES")
        logger.info("="*80)

        # Predict median survival time (expected time to repurchase)
        predictions = self.model.predict_median(df.drop(['duration', 'event_observed'], axis=1))

        # Add predictions to original dataframe
        # Cap predictions at 730 days (2 years) to avoid overflow and keep business-relevant
        original_df['predicted_days_to_repurchase'] = predictions.values
        original_df['predicted_days_to_repurchase'] = original_df['predicted_days_to_repurchase'].clip(upper=730)

        # Calculate expected reorder date (handling potential None values)
        try:
            original_df['expected_reorder_date'] = pd.Timestamp.now() + pd.to_timedelta(
                original_df['predicted_days_to_repurchase'], unit='D'
            )
        except Exception as e:
            logger.warning(f"Error calculating expected_reorder_date: {e}")
            # Set to null if calculation fails
            original_df['expected_reorder_date'] = pd.NaT

        # Predict survival probability at different time points
        # Probability of NOT repurchasing within X days
        time_points = [7, 14, 30, 60, 90]

        for days in time_points:
            survival_probs = self.model.predict_survival_function(
                df.drop(['duration', 'event_observed'], axis=1),
                times=[days]
            ).T.values.flatten()

            # Reorder probability = 1 - survival probability
            original_df[f'reorder_prob_{days}d'] = 1 - survival_probs

        logger.info(f"✓ Predicted repurchase times for {len(original_df):,} customer-product pairs")
        logger.info(f"\nMedian predicted days to repurchase: {predictions.median():.1f}")
        logger.info(f"Mean predicted days to repurchase: {predictions.mean():.1f}")

        return original_df

    def generate_reorder_alerts(self, predictions_df):
        """Generate proactive reorder alerts"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING REORDER ALERTS")
        logger.info("="*80)

        # Define alert criteria
        # High priority: Should have reordered by now
        # Medium priority: Will reorder soon
        # Low priority: Recently purchased

        alerts = []

        for _, row in predictions_df.iterrows():
            days_since = row['days_since_last_purchase']
            expected_days = row['predicted_days_to_repurchase']

            # Calculate how overdue they are
            days_overdue = days_since - expected_days

            # Priority logic
            if days_overdue > 30:
                priority = 'HIGH'
                message = f"URGENT: Customer is {days_overdue:.0f} days overdue for repurchase"
            elif days_overdue > 0:
                priority = 'MEDIUM'
                message = f"Customer is {days_overdue:.0f} days overdue"
            elif row['reorder_prob_30d'] > 0.7:
                priority = 'MEDIUM'
                message = f"High probability ({row['reorder_prob_30d']:.0%}) of reorder in next 30 days"
            elif row['reorder_prob_60d'] > 0.5:
                priority = 'LOW'
                message = f"Moderate probability ({row['reorder_prob_60d']:.0%}) of reorder in next 60 days"
            else:
                continue  # Skip low-probability alerts

            alerts.append({
                'customer_id': row['customer_id'],
                'product_id': row['product_id'],
                'priority': priority,
                'days_since_last_purchase': days_since,
                'predicted_days_to_repurchase': expected_days,
                'days_overdue': days_overdue,
                'reorder_prob_30d': row['reorder_prob_30d'],
                'expected_reorder_date': row['expected_reorder_date'],
                'message': message,
                'customer_tier': row.get('customer_tier', 'Unknown'),
                'revenue_potential': row.get('revenue_potential', 'Unknown')
            })

        alerts_df = pd.DataFrame(alerts)

        if len(alerts_df) > 0:
            # Sort by priority and days overdue
            priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            alerts_df['priority_rank'] = alerts_df['priority'].map(priority_order)
            alerts_df = alerts_df.sort_values(['priority_rank', 'days_overdue'], ascending=[True, False])
            alerts_df = alerts_df.drop('priority_rank', axis=1)

            logger.info(f"✓ Generated {len(alerts_df):,} reorder alerts")
            logger.info(f"\nAlert breakdown:")
            logger.info(f"  HIGH priority: {(alerts_df['priority'] == 'HIGH').sum():,}")
            logger.info(f"  MEDIUM priority: {(alerts_df['priority'] == 'MEDIUM').sum():,}")
            logger.info(f"  LOW priority: {(alerts_df['priority'] == 'LOW').sum():,}")

            # Show top 10 alerts
            logger.info(f"\nTop 10 HIGH Priority Alerts:")
            high_priority = alerts_df[alerts_df['priority'] == 'HIGH'].head(10)
            for i, (_, alert) in enumerate(high_priority.iterrows(), 1):
                logger.info(f"  {i}. Customer {alert['customer_id']} - Product {alert['product_id']}")
                logger.info(f"     {alert['message']}")
                logger.info(f"     Tier: {alert['customer_tier']}, Revenue potential: {alert['revenue_potential']}")
        else:
            logger.info("No alerts generated (all customers on track)")

        return alerts_df

    def save_model(self, predictions_df, alerts_df):
        """Save trained model and predictions"""
        logger.info("\n" + "="*80)
        logger.info("SAVING MODEL")
        logger.info("="*80)

        # Save model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"✓ Model saved to {MODEL_PATH}")

        # Save alerts
        if len(alerts_df) > 0:
            alerts_df.to_csv(ALERTS_PATH, index=False)
            logger.info(f"✓ Alerts saved to {ALERTS_PATH}")

        # Save predictions summary
        summary_path = MODEL_DIR / "repurchase_predictions.csv"
        predictions_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Predictions saved to {summary_path}")

    def run(self):
        """Main training pipeline"""
        try:
            start_time = datetime.now()
            logger.info("\n" + "="*80)
            logger.info("SURVIVAL ANALYSIS - REPURCHASE PREDICTION")
            logger.info("="*80)
            logger.info(f"Start time: {start_time}")
            logger.info(f"Goal: Predict WHEN customers will reorder (win-back opportunity)")

            # Load data
            df = self.load_data()

            # Prepare training data
            train_df, original_df = self.prepare_training_data(df)

            # Train model
            self.train(train_df)

            # Evaluate
            self.evaluate(train_df)

            # Predict repurchase times
            predictions_df = self.predict_repurchase_times(train_df, original_df)

            # Generate alerts
            alerts_df = self.generate_reorder_alerts(predictions_df)

            # Save
            self.save_model(predictions_df, alerts_df)

            # Summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("\n" + "="*80)
            logger.info("✅ SURVIVAL ANALYSIS COMPLETE!")
            logger.info("="*80)
            logger.info(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info(f"\nModel files:")
            logger.info(f"  {MODEL_PATH}")
            logger.info(f"  {ALERTS_PATH}")
            logger.info(f"\nBusiness Impact:")
            if len(alerts_df) > 0:
                high_priority = len(alerts_df[alerts_df['priority'] == 'HIGH'])
                logger.info(f"  {high_priority:,} HIGH priority customers need immediate contact")
                logger.info(f"  Estimated revenue opportunity: ${high_priority * 15000:,} (at $15K/customer)")
            logger.info(f"\nNext step: Integrate with CRM for proactive outreach!")

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    analyzer = RepurchaseSurvivalAnalysis()
    success = analyzer.run()
    sys.exit(0 if success else 1)
