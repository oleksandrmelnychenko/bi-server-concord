#!/bin/bash
# Auto-validate GNN v2 and Ensemble v2 when training completes

echo "Waiting for GNN training to complete..."

# Wait for training to finish (check every 60 seconds)
while true; do
    # Check if training log shows completion
    if tail -5 gnn_training_v2.log | grep -q "Training complete"; then
        echo "Training complete! Starting validation..."
        break
    fi

    # Check if process is still running
    if ! ps aux | grep -v grep | grep -q "train_gnn_recommender.py"; then
        echo "Training process completed or stopped. Starting validation..."
        break
    fi

    sleep 60
done

sleep 5

echo "================================================================================"
echo "PHASE C: VALIDATING RETRAINED GNN AND ENSEMBLE"
echo "================================================================================"
echo ""

# Validate GNN v2
echo "Step 1: Validating GNN v2 (retrained model)..."
python3 scripts/validate_gnn_recommender.py 2>&1 | tee gnn_validation_v2.log

echo ""
echo "Step 2: Validating Ensemble v2 (70/30 with retrained GNN)..."
python3 scripts/ensemble_recommender.py 2>&1 | tee ensemble_validation_v2.log

echo ""
echo "================================================================================"
echo "VALIDATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - GNN v2: results/gnn_validation_results.csv"
echo "  - Ensemble v2: results/ensemble_validation_results.csv"
echo ""
echo "Logs:"
echo "  - GNN v2: gnn_validation_v2.log"
echo "  - Ensemble v2: ensemble_validation_v2.log"
echo ""
