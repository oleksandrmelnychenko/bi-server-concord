# Project Cleanup Report
**Generated**: 2025-11-12
**Total Files Analyzed**: 38 Python files + 18 Markdown files

---

## Executive Summary

**Cleanup Opportunity**: 22 Python files (58%) and 4 Markdown files can be safely removed or archived.

**Storage Impact**:
- Python files for removal: ~22 files
- Test/diagnostic files: ~15 files
- Old documentation: ~4 markdown files

**Recommendation**: DELETE old versions, ARCHIVE test scripts, KEEP production and documentation

---

## Python Files Analysis (38 total)

###  KEEP - Production Files (16 files - 42%)

#### API Layer (5 files)
- `api/main.py` - API server entry point
- `api/db_pool.py` - Database connection pool
- `api/routes/recommendations.py` - Recommendation API routes
- `api/models/forecast_schemas.py` - Forecast data models
- `api/models/recommendation_schemas.py` - Recommendation data models

#### Workers (2 files)
- `scripts/forecasting/forecast_worker.py` - Production forecast worker
- `scripts/weekly_recommendation_worker.py` - Production recommendation worker **[ACTIVE]**

#### Core Algorithms (8 files)
- `scripts/improved_hybrid_recommender_v33.py` - **V3.3 Recommender [CURRENT VERSION - RECENTLY FIXED]**
- `scripts/forecasting/core/pattern_analyzer.py` - Pattern analysis
- `scripts/forecasting/core/customer_predictor.py` - Customer predictions
- `scripts/forecasting/core/product_aggregator.py` - Product aggregation
- `scripts/forecasting/core/forecast_engine.py` - Forecast engine core
- `scripts/forecasting/__init__.py` - Package init
- `scripts/forecasting/core/__init__.py` - Core package init

#### Utilities (2 files)
- `scripts/datetime_utils.py` - Date/time utilities
- `scripts/redis_helper.py` - Redis helper functions

---

### DELETE - Old Recommender Versions (5 files - SAFE TO DELETE)

These have been superseded by V3.3 (the current production version):

1. **scripts/improved_hybrid_recommender.py**
   Original version, superseded by v3.1

2. **scripts/improved_hybrid_recommender_v31.py**
   Version 3.1, superseded by v3.2

3. **scripts/improved_hybrid_recommender_v32.py**
   Version 3.2, superseded by v3.3

4. **scripts/improved_hybrid_recommender_v34.py**
   Experimental version 3.4, never deployed

5. **scripts/improved_hybrid_recommender_v4.py**
   **FAILED VERSION** - Brand/category features failed testing
   Documented failure in `/tmp/v4_final_findings.md`
   Precision regression: 19.7% → 17.35% (-2.35%)

---

### ARCHIVE - Test & Diagnostic Scripts (15 files)

#### Category: Testing & Validation (5 files)
- `test_forecast.py` - Ad-hoc forecast testing
- `scripts/forecasting/test_worker.py` - Worker testing
- `scripts/forecasting/test_rfm_accuracy.py` - RFM accuracy testing
- `scripts/validate_precision.py` - Precision validation
- `scripts/debug_validation.py` - Validation debugging

#### Category: Analysis & Diagnostics (6 files)
- `scripts/analyze_feature_effectiveness.py` - Feature analysis
- `scripts/diagnostic_deep_dive.py` - Deep diagnostics
- `scripts/explore_product_categories.py` - Category exploration
- `scripts/inspect_product_groups.py` - Product group inspection
- `scripts/deep_test_v33_and_v4_hypotheses.py` - V3.3 vs V4 hypothesis testing
- `scripts/comprehensive_v4_diagnostic.py` - V4 comprehensive diagnostics

#### Category: Optimization Experiments (2 files)
- `scripts/tune_hyperparameters.py` - Hyperparameter tuning
- `scripts/grid_search_weights.py` - Weight grid search

#### Category: One-time Utilities (2 files)
- `remove_comments.py` - One-time comment removal script
- `remove_redundant_comments.py` - One-time redundant comment removal

---

### INVESTIGATE - Potential Issues (2 files)

1. **workers/weekly_recommendation_worker.py**
   **Status**: Files `workers/weekly_recommendation_worker.py` and `scripts/weekly_recommendation_worker.py` are DIFFERENT
   **Action Required**: Determine which is correct, delete the other
   **Current Usage**: `scripts/weekly_recommendation_worker.py` is referenced in `start_workers.sh`

2. **scripts/forecasting/trigger_relearn.py**
   **Status**: Manual trigger utility
   **Action**: KEEP if useful for manual operations, otherwise archive

---

## Markdown Documentation Analysis (18 total)

### KEEP - Core Documentation (14 files)

#### Project Documentation (4 files)
- `README.md` - Project README
- `DEPLOYMENT.md` - Deployment guide
- `DOCKER_DEPLOYMENT.md` - Docker deployment
- `sample_forecasts/README.md` - Sample forecasts README

#### Algorithm Documentation (10 files)
- `docs/algorithms/CUSTOMER_PREDICTOR.md` - Customer predictor docs
- `docs/algorithms/PATTERN_ANALYZER.md` - Pattern analyzer docs
- `docs/algorithms/HYBRID_RECOMMENDER_V32.md` - V3.2 recommender docs
- `docs/algorithms/PRODUCT_AGGREGATOR.md` - Product aggregator docs
- `docs/algorithms/FORECAST_ENGINE.md` - Forecast engine docs
- `docs/utilities/DATETIME_UTILITIES.md` - Datetime utilities docs
- `docs/api/API_ENDPOINTS.md` - API endpoints docs
- `docs/workers/FORECAST_WORKER.md` - Forecast worker docs
- `docs/workers/RECOMMENDATION_WORKER.md` - Recommendation worker docs

---

### ARCHIVE - Old Experiment Documentation (4 files)

These document completed experiments and can be moved to `/archive`:

1. **DATETIME_FIXES_SUMMARY.md**
   Historical datetime fixes documentation

2. **TECHNOLOGY_STACK_PROPOSAL.md**
   Technology stack proposal (likely outdated)

3. **TRACK1_QUICK_WINS_RESULTS.md**
   Track 1 quick wins results

4. **HYPERPARAMETER_TUNING_RESULTS.md**
   Hyperparameter tuning results

5. **ULTRA_DEEP_ANALYSIS_REPORT.md**
   Ultra-deep analysis report

---

## Cleanup Action Plan

### Phase 1: DELETE Old Versions (Immediate - Low Risk)

```bash
# Delete old recommender versions
rm scripts/improved_hybrid_recommender.py
rm scripts/improved_hybrid_recommender_v31.py
rm scripts/improved_hybrid_recommender_v32.py
rm scripts/improved_hybrid_recommender_v34.py
rm scripts/improved_hybrid_recommender_v4.py

# Delete one-time utility scripts
rm remove_comments.py
rm remove_redundant_comments.py
```

**Risk**: None - these files are superseded
**Storage Savings**: ~7 files

---

### Phase 2: ARCHIVE Test Scripts (Recommended)

Create archive directory and move test files:

```bash
# Create archive directories
mkdir -p archive/test_scripts
mkdir -p archive/diagnostics
mkdir -p archive/docs

# Move test scripts
mv test_forecast.py archive/test_scripts/
mv scripts/forecasting/test_*.py archive/test_scripts/
mv scripts/validate_precision.py archive/test_scripts/
mv scripts/debug_validation.py archive/test_scripts/

# Move diagnostic scripts
mv scripts/analyze_feature_effectiveness.py archive/diagnostics/
mv scripts/diagnostic_deep_dive.py archive/diagnostics/
mv scripts/explore_product_categories.py archive/diagnostics/
mv scripts/inspect_product_groups.py archive/diagnostics/
mv scripts/deep_test_v33_and_v4_hypotheses.py archive/diagnostics/
mv scripts/comprehensive_v4_diagnostic.py archive/diagnostics/

# Move optimization experiments
mv scripts/tune_hyperparameters.py archive/diagnostics/
mv scripts/grid_search_weights.py archive/diagnostics/

# Move old documentation
mv DATETIME_FIXES_SUMMARY.md archive/docs/
mv TECHNOLOGY_STACK_PROPOSAL.md archive/docs/
mv TRACK1_QUICK_WINS_RESULTS.md archive/docs/
mv HYPERPARAMETER_TUNING_RESULTS.md archive/docs/
mv ULTRA_DEEP_ANALYSIS_REPORT.md archive/docs/
```

**Risk**: Low - files are preserved for historical reference
**Storage Impact**: ~19 files moved to archive

---

### Phase 3: RESOLVE Duplicate Worker (Investigation Required)

**Task**: Determine which worker file is correct

```bash
# Check which file is currently used
grep -r "weekly_recommendation_worker" start_workers.sh stop_workers.sh

# Compare the two files
diff scripts/weekly_recommendation_worker.py workers/weekly_recommendation_worker.py

# After determining the correct file:
# Option A: Delete workers/weekly_recommendation_worker.py
# Option B: Delete scripts/weekly_recommendation_worker.py and update scripts
```

**Action Required**: Manual investigation before deletion

---

### Phase 4: Optional - Keep Utility Script

**Decision Point**: `scripts/forecasting/trigger_relearn.py`

- **KEEP** if you manually trigger re-learning
- **ARCHIVE** if you never use it

---

## Summary Statistics

### Before Cleanup
- Total Python files: 38
- Total Markdown files: 18
- **Total**: 56 files

### After Cleanup (Recommended)
- Production Python files: 16
- Core documentation: 14
- Archived files: 24
- Deleted files: 7
- **Total Active**: 30 files (-46% reduction)

---

## Impact Analysis

### Storage Impact
- **Deleted**: ~7 files (old versions, one-time utilities)
- **Archived**: ~24 files (tests, diagnostics, old docs)
- **Reduction**: 46% fewer active files in main directories

### Maintenance Impact
- Clearer project structure
- Easier to navigate codebase
- Reduced confusion about which version to use
- Historical files preserved in archive for reference

### Risk Assessment
- **LOW RISK**: Old recommender versions are superseded
- **ZERO RISK**: Test scripts can be recovered from archive
- **INVESTIGATION NEEDED**: Duplicate worker file

---

## Current Production Stack (KEEP THESE)

**API Server**: `api/main.py`
**Database**: `api/db_pool.py`
**Recommender**: `scripts/improved_hybrid_recommender_v33.py` (V3.3 - ACTIVE, recently fixed)
**Forecast Worker**: `scripts/forecasting/forecast_worker.py`
**Recommendation Worker**: `scripts/weekly_recommendation_worker.py`
**Core Algorithms**: All files in `scripts/forecasting/core/`

---

## Notes

1. **V3.3 is the current production version** - recently fixed to enable collaborative filtering
2. **V4 failed testing** - documented in /tmp/v4_final_findings.md (precision regression)
3. **Workers are managed** via `start_workers.sh`, `stop_workers.sh`, `check_workers.sh`
4. **Redis is required** for caching (forecast and recommendation systems)

---

## Next Steps

1. Review this report
2. Execute Phase 1 (DELETE old versions) - safest cleanup
3. Execute Phase 2 (ARCHIVE test scripts) - recommended
4. Investigate Phase 3 (duplicate worker) - requires manual review
5. Decide on Phase 4 (trigger_relearn.py) - based on usage

**Estimated Time**: 15-30 minutes for full cleanup
