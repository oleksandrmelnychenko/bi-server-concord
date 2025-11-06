"""
Sales Forecasting Model Training

Trains a Prophet model for time series forecasting of sales data.
"""
import os
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
import mlflow.pyfunc
import joblib
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastingModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for Prophet forecasting model"""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        """
        Generate sales forecast

        Args:
            model_input: DataFrame with 'periods' column (number of days to forecast)

        Returns:
            DataFrame with forecast
        """
        if 'periods' not in model_input.columns:
            periods = 30  # Default
        else:
            periods = int(model_input['periods'].iloc[0])

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)

        # Generate forecast
        forecast = self.model.predict(future)

        # Return only future predictions
        forecast = forecast.tail(periods)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def load_sales_data() -> pd.DataFrame:
    """
    Load historical sales data

    Returns:
        DataFrame with ds (date) and y (sales) columns
    """
    # TODO: Load from PostgreSQL/Delta Lake
    # For now, generate synthetic data

    logger.info("Loading sales data...")

    # Generate synthetic daily sales data for 2 years
    np.random.seed(42)

    start_date = datetime.now() - timedelta(days=730)
    dates = pd.date_range(start=start_date, periods=730, freq='D')

    # Base trend + seasonality + noise
    trend = np.linspace(10000, 15000, 730)
    seasonal = 2000 * np.sin(2 * np.pi * np.arange(730) / 365)  # Yearly seasonality
    weekly = 500 * np.sin(2 * np.pi * np.arange(730) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 500, 730)

    sales = trend + seasonal + weekly + noise

    df = pd.DataFrame({
        'ds': dates,
        'y': sales
    })

    logger.info(f"Loaded {len(df)} days of sales data")

    return df


def create_train_test_split(
    data: pd.DataFrame,
    test_days: int = 90
) -> tuple:
    """
    Split data into train and test sets

    Args:
        data: Sales data
        test_days: Number of days for test set

    Returns:
        Train and test DataFrames
    """
    split_date = data['ds'].max() - timedelta(days=test_days)

    train = data[data['ds'] <= split_date]
    test = data[data['ds'] > split_date]

    logger.info(f"Train: {len(train)} days, Test: {len(test)} days")

    return train, test


def train_model(
    train_data: pd.DataFrame,
    seasonality_mode: str = 'additive',
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0
) -> Prophet:
    """
    Train Prophet forecasting model

    Args:
        train_data: Training data with ds and y columns
        seasonality_mode: 'additive' or 'multiplicative'
        changepoint_prior_scale: Flexibility of trend changes
        seasonality_prior_scale: Strength of seasonality

    Returns:
        Trained Prophet model
    """
    logger.info("Training forecasting model...")

    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    # Add custom seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # Train
    model.fit(train_data)

    logger.info("Training complete!")

    return model


def evaluate_model(
    model: Prophet,
    test_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Evaluate model performance

    Args:
        model: Trained model
        test_data: Test data

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")

    # Generate predictions for test period
    forecast = model.predict(test_data[['ds']])

    # Calculate metrics
    y_true = test_data['y'].values
    y_pred = forecast['yhat'].values

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Coverage (percentage of actual values within prediction intervals)
    within_interval = np.sum(
        (y_true >= forecast['yhat_lower'].values) &
        (y_true <= forecast['yhat_upper'].values)
    )
    coverage = within_interval / len(y_true) * 100

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'coverage': coverage
    }

    logger.info(f"Evaluation metrics: {metrics}")

    return metrics


def perform_cross_validation(model: Prophet, initial_days: int = 545) -> pd.DataFrame:
    """
    Perform time series cross-validation

    Args:
        model: Trained model
        initial_days: Initial training period

    Returns:
        Cross-validation results
    """
    logger.info("Performing cross-validation...")

    df_cv = cross_validation(
        model,
        initial=f'{initial_days} days',
        period='90 days',
        horizon='90 days'
    )

    df_metrics = performance_metrics(df_cv)

    logger.info(f"CV metrics:\n{df_metrics.describe()}")

    return df_metrics


def main():
    """Main training pipeline"""

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))

    # Create experiment
    experiment_name = "sales-forecasting"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # Log parameters
        params = {
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'test_days': 90
        }
        mlflow.log_params(params)

        # Load data
        data = load_sales_data()
        mlflow.log_metric('total_days', len(data))

        # Split data
        train, test = create_train_test_split(data, params['test_days'])

        # Train model
        model = train_model(
            train,
            seasonality_mode=params['seasonality_mode'],
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale']
        )

        # Evaluate
        metrics = evaluate_model(model, test)
        mlflow.log_metrics(metrics)

        # Cross-validation
        cv_metrics = perform_cross_validation(model)
        mlflow.log_metrics({
            'cv_mae': cv_metrics['mae'].mean(),
            'cv_rmse': cv_metrics['rmse'].mean(),
            'cv_mape': cv_metrics['mape'].mean()
        })

        # Save model
        wrapped_model = ForecastingModelWrapper(model)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped_model,
            pip_requirements=[
                "prophet==1.1.5",
                "numpy==1.26.2",
                "pandas==2.1.4"
            ]
        )

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_details = mlflow.register_model(model_uri, "sales-forecasting")

        logger.info(f"Model registered: {model_details.name} version {model_details.version}")

        print("\n=== Training Complete ===")
        print(f"Model: {model_details.name}")
        print(f"Version: {model_details.version}")
        print(f"Metrics:")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  Coverage: {metrics['coverage']:.2f}%")
        print(f"MLflow UI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
