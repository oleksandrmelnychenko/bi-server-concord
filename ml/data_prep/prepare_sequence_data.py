"""
Prepare Purchase Sequence Datasets for LSTM + Weibull Repurchase Model

This script creates time-ordered purchase sequences for each customer,
which will be used to train the LSTM component of the repurchase prediction model.

Output Format:
- Customer-level sequences with temporal features
- Inter-purchase time intervals
- Product purchase history with timestamps
- Saved as PyTorch-compatible .pt files
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import torch
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DUCKDB_PATH = "/opt/dagster/app/data/dbt/concord_bi.duckdb"
OUTPUT_DIR = "/opt/dagster/app/data/processed/sequences"

class SequenceDataPreparator:
    """Prepare purchase sequence data for temporal models"""

    def __init__(self, duckdb_path: str, output_dir: str):
        self.duckdb_path = duckdb_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_order_data(self) -> pd.DataFrame:
        """
        Load order-level data with timestamps

        Returns:
            DataFrame with columns: customer_id, product_id, order_date, quantity, amount
        """
        logger.info("Loading order data from DuckDB...")

        conn = duckdb.connect(self.duckdb_path, read_only=True)

        # Get order history from source tables
        query = """
        SELECT
            o.CustomerID as customer_id,
            od.ProductID as product_id,
            o.OrderDate as order_date,
            od.Quantity as quantity,
            od.LineTotal as amount,
            p.ProductName as product_name
        FROM raw.orders o
        JOIN raw.order_details od ON o.OrderID = od.OrderID
        JOIN raw.products p ON od.ProductID = p.ProductID
        WHERE o.OrderDate IS NOT NULL
            AND o.CustomerID IS NOT NULL
            AND od.ProductID IS NOT NULL
        ORDER BY o.CustomerID, o.OrderDate, od.ProductID
        """

        df = conn.execute(query).df()
        conn.close()

        # Convert date column to datetime
        df['order_date'] = pd.to_datetime(df['order_date'])

        logger.info(f"Loaded {len(df):,} order records for {df['customer_id'].nunique():,} customers")

        return df

    def create_customer_sequences(self, orders_df: pd.DataFrame) -> Dict:
        """
        Create purchase sequences for each customer

        Args:
            orders_df: DataFrame with order history

        Returns:
            Dict with customer sequences and metadata
        """
        logger.info("Creating customer purchase sequences...")

        sequences = {}

        for customer_id, customer_orders in orders_df.groupby('customer_id'):
            # Sort by date
            customer_orders = customer_orders.sort_values('order_date')

            # Calculate inter-purchase intervals (days)
            order_dates = customer_orders['order_date'].unique()
            order_dates_sorted = np.sort(order_dates)

            if len(order_dates_sorted) < 2:
                # Skip customers with only 1 purchase (can't calculate intervals)
                continue

            # Inter-purchase intervals in days
            intervals = np.diff([d.timestamp() for d in order_dates_sorted]) / (24 * 3600)

            # Create sequence of products purchased
            product_sequence = []
            quantity_sequence = []
            amount_sequence = []
            date_sequence = []

            for _, order in customer_orders.iterrows():
                product_sequence.append(str(order['product_id']))
                quantity_sequence.append(float(order['quantity']))
                amount_sequence.append(float(order['amount']))
                date_sequence.append(order['order_date'])

            # Calculate recency (days since last purchase)
            last_purchase_date = customer_orders['order_date'].max()
            recency_days = (datetime.now() - last_purchase_date).days

            sequences[str(customer_id)] = {
                'product_ids': product_sequence,
                'quantities': quantity_sequence,
                'amounts': amount_sequence,
                'dates': date_sequence,
                'inter_purchase_intervals': intervals.tolist(),
                'num_purchases': len(customer_orders),
                'num_unique_products': customer_orders['product_id'].nunique(),
                'first_purchase_date': customer_orders['order_date'].min(),
                'last_purchase_date': last_purchase_date,
                'recency_days': recency_days,
                'total_spent': float(customer_orders['amount'].sum()),
                'avg_order_value': float(customer_orders['amount'].mean()),
                'purchase_frequency': len(order_dates_sorted) / ((order_dates_sorted[-1] - order_dates_sorted[0]).days / 365.25) if len(order_dates_sorted) > 1 else 0
            }

        logger.info(f"Created sequences for {len(sequences):,} customers with 2+ purchases")

        return sequences

    def create_product_sequences_by_product(self, orders_df: pd.DataFrame, sequences: Dict) -> Dict:
        """
        Create per-product purchase sequences (for repurchase prediction)

        For each customer-product pair, track:
        - When they bought this product
        - How much time between repurchases
        - Context (what else they were buying)

        Args:
            orders_df: DataFrame with order history
            sequences: Customer sequences from create_customer_sequences

        Returns:
            Dict with product-level repurchase data
        """
        logger.info("Creating product-level repurchase sequences...")

        product_sequences = {}

        for customer_id, customer_orders in orders_df.groupby('customer_id'):
            customer_id_str = str(customer_id)

            if customer_id_str not in sequences:
                continue  # Skip customers with <2 purchases

            for product_id, product_orders in customer_orders.groupby('product_id'):
                product_id_str = str(product_id)

                # Sort by date
                product_orders = product_orders.sort_values('order_date')

                if len(product_orders) < 2:
                    # Need at least 2 purchases to predict repurchase
                    continue

                # Calculate time-to-next-purchase for each purchase event
                purchase_dates = product_orders['order_date'].values
                quantities = product_orders['quantity'].values
                amounts = product_orders['amount'].values

                # Time to next purchase (in days)
                time_to_next = []
                for i in range(len(purchase_dates) - 1):
                    days_to_next = (purchase_dates[i+1] - purchase_dates[i]) / np.timedelta64(1, 'D')
                    time_to_next.append(float(days_to_next))

                key = f"{customer_id_str}_{product_id_str}"

                product_sequences[key] = {
                    'customer_id': customer_id_str,
                    'product_id': product_id_str,
                    'purchase_dates': [d.isoformat() for d in purchase_dates],
                    'quantities': quantities.tolist(),
                    'amounts': amounts.tolist(),
                    'time_to_next_purchase': time_to_next,  # Target for training
                    'num_repurchases': len(purchase_dates),
                    'avg_inter_purchase_time': np.mean(time_to_next) if time_to_next else None,
                    'std_inter_purchase_time': np.std(time_to_next) if len(time_to_next) > 1 else None,
                    'last_purchase_date': purchase_dates[-1].isoformat(),
                    'days_since_last_purchase': (datetime.now() - pd.to_datetime(purchase_dates[-1])).days
                }

        logger.info(f"Created repurchase sequences for {len(product_sequences):,} customer-product pairs")

        return product_sequences

    def create_train_test_split(self, product_sequences: Dict, test_ratio: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Split product sequences into train/test sets

        Strategy: Hold out last purchase event for each customer-product pair as test

        Args:
            product_sequences: Product-level sequences
            test_ratio: Fraction of sequences to hold out for testing

        Returns:
            (train_sequences, test_sequences)
        """
        logger.info(f"Splitting data into train/test (test_ratio={test_ratio})...")

        # Sort sequences by customer_id for consistent splitting
        sorted_keys = sorted(product_sequences.keys())

        # Shuffle with fixed seed for reproducibility
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(sorted_keys))

        split_idx = int(len(sorted_keys) * (1 - test_ratio))
        train_indices = shuffled_indices[:split_idx]
        test_indices = shuffled_indices[split_idx:]

        train_sequences = {sorted_keys[i]: product_sequences[sorted_keys[i]] for i in train_indices}
        test_sequences = {sorted_keys[i]: product_sequences[sorted_keys[i]] for i in test_indices}

        logger.info(f"Train: {len(train_sequences):,} sequences | Test: {len(test_sequences):,} sequences")

        return train_sequences, test_sequences

    def save_sequences(self, sequences: Dict, product_sequences: Dict,
                       train_sequences: Dict, test_sequences: Dict):
        """Save all sequence data to disk"""
        logger.info(f"Saving sequences to {self.output_dir}...")

        # Save as pickle files
        with open(self.output_dir / 'customer_sequences.pkl', 'wb') as f:
            pickle.dump(sequences, f)

        with open(self.output_dir / 'product_repurchase_sequences.pkl', 'wb') as f:
            pickle.dump(product_sequences, f)

        with open(self.output_dir / 'train_sequences.pkl', 'wb') as f:
            pickle.dump(train_sequences, f)

        with open(self.output_dir / 'test_sequences.pkl', 'wb') as f:
            pickle.dump(test_sequences, f)

        # Save metadata
        metadata = {
            'num_customers': len(sequences),
            'num_customer_product_pairs': len(product_sequences),
            'train_size': len(train_sequences),
            'test_size': len(test_sequences),
            'created_at': datetime.now().isoformat(),
            'description': 'Purchase sequence data for LSTM + Weibull repurchase model'
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            import json
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Saved all sequences successfully")
        logger.info(f"  - Customer sequences: {len(sequences):,}")
        logger.info(f"  - Product repurchase pairs: {len(product_sequences):,}")
        logger.info(f"  - Train: {len(train_sequences):,} | Test: {len(test_sequences):,}")

    def run(self):
        """Main execution"""
        logger.info("="*60)
        logger.info("PURCHASE SEQUENCE DATA PREPARATION")
        logger.info("="*60)

        # Load data
        orders_df = self.load_order_data()

        # Create customer sequences
        customer_sequences = self.create_customer_sequences(orders_df)

        # Create product-level repurchase sequences
        product_sequences = self.create_product_sequences_by_product(orders_df, customer_sequences)

        # Train/test split
        train_seq, test_seq = self.create_train_test_split(product_sequences, test_ratio=0.2)

        # Save everything
        self.save_sequences(customer_sequences, product_sequences, train_seq, test_seq)

        logger.info("="*60)
        logger.info("SEQUENCE PREPARATION COMPLETE!")
        logger.info("="*60)

        return {
            'customer_sequences': customer_sequences,
            'product_sequences': product_sequences,
            'train_sequences': train_seq,
            'test_sequences': test_seq
        }

def main():
    """Entry point"""
    preparator = SequenceDataPreparator(DUCKDB_PATH, OUTPUT_DIR)
    result = preparator.run()

    return result

if __name__ == "__main__":
    main()
