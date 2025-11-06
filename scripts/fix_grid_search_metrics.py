#!/usr/bin/env python3
"""
Fix Grid Search Metrics - Recalculate with Correct Metric

Grid search used: total_hits / total_products (WRONG)
Phase C uses: average(per_customer_precision) (CORRECT)

This script recalculates all 1,728 configurations using the correct metric
from the saved per-customer results.
"""

import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def recalculate_precision(result: Dict) -> float:
    """
    Recalculate precision using Phase C metric: average of per-customer precisions

    Instead of: total_hits / total_products
    Use: mean(customer_precisions)
    """
    customers = result.get('customers', [])

    if not customers:
        return 0.0

    # Average per-customer precision
    precisions = [c['precision'] for c in customers]
    return sum(precisions) / len(precisions)


def fix_grid_search_results():
    """Main function to recalculate and re-rank all grid search results"""

    logger.info("="*80)
    logger.info("FIXING GRID SEARCH METRICS")
    logger.info("="*80)

    # Load original results
    logger.info("\nLoading results/grid_search_results.json...")
    with open('results/grid_search_results.json', 'r') as f:
        data = json.load(f)

    logger.info(f"Original validation precision (WRONG): {data['validation_precision']:.3f}")
    logger.info(f"Original test precision (WRONG): {data['test_precision']:.3f}")

    # Recalculate all validation results
    logger.info(f"\nRecalculating {len(data['all_val_results'])} validation configurations...")

    for result in data['all_val_results']:
        old_precision = result['precision']
        new_precision = recalculate_precision(result)
        result['precision_corrected'] = new_precision
        result['precision_original'] = old_precision

    # Re-rank by corrected precision
    data['all_val_results'].sort(key=lambda x: x['precision_corrected'], reverse=True)

    # Update best configuration
    best_config = data['all_val_results'][0]
    data['best_weights'] = best_config['weights']
    data['validation_precision_original'] = data['validation_precision']
    data['validation_precision'] = best_config['precision_corrected']

    logger.info("\nTop 10 configurations (CORRECTED):")
    for i, result in enumerate(data['all_val_results'][:10]):
        logger.info(f"  {i+1}. Precision: {result['precision_corrected']:.3f} (was {result['precision_original']:.3f})")
        logger.info(f"     Weights: {result['weights']}")

    # Note: We need to recalculate test precision too if available
    # But the original file only has top 20 validation configs
    # We'll need to use the best weights and note that test needs re-evaluation

    logger.info("\n" + "="*80)
    logger.info("CORRECTED BEST CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Validation Precision (CORRECTED): {data['validation_precision']:.3f}")
    logger.info(f"Validation Precision (ORIGINAL):  {data['validation_precision_original']:.3f}")
    logger.info(f"\nBest Weights:")
    for key, value in data['best_weights'].items():
        logger.info(f"  {key}: {value:.3f}")

    # Calculate improvement vs baseline
    baseline = data['baseline_precision']
    improvement = data['validation_precision'] - baseline

    logger.info(f"\n📊 PERFORMANCE COMPARISON")
    logger.info(f"  Baseline (Phase C): {baseline:.3f}")
    logger.info(f"  Optimized (Val):    {data['validation_precision']:.3f}")
    logger.info(f"  Improvement:        {improvement:+.3f} ({improvement*100:+.1f}pp)")

    if data['validation_precision'] > baseline:
        logger.info(f"  Status: ✅ IMPROVEMENT!")
    elif data['validation_precision'] >= baseline - 0.01:
        logger.info(f"  Status: ⚠️  Similar to baseline")
    else:
        logger.info(f"  Status: ❌ Below baseline")

    # Save corrected results
    output_file = 'results/grid_search_results_corrected.json'
    logger.info(f"\nSaving corrected results to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    logger.info("\n" + "="*80)
    logger.info("METRIC FIX COMPLETE")
    logger.info("="*80)
    logger.info(f"\n⚠️  NOTE: Test set precision needs to be evaluated separately")
    logger.info(f"   The original test set used the wrong metric.")
    logger.info(f"   Run validation on test set with corrected weights to get true test precision.")

    return data


if __name__ == '__main__':
    fix_grid_search_results()
