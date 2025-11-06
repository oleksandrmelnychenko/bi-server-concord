#!/usr/bin/env python3
"""
Daily Model Refresh Pipeline

Runs every day at 2 AM to:
1. Extract fresh data from MSSQL
2. Retrain Collaborative Filtering model
3. Retrain Survival Analysis model
4. Generate fresh reorder alerts

Total time: ~20 seconds

Setup cron job:
    0 2 * * * cd /path/to/Concord-BI-Server && python3 scripts/refresh_models_daily.py >> logs/daily_refresh.log 2>&1
"""

import subprocess
import logging
import sys
from datetime import datetime
from pathlib import Path
import smtplib
from email.mime.text import MIMEText

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"daily_refresh_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DailyRefreshPipeline:
    """Automated daily model retraining pipeline"""

    def __init__(self):
        self.start_time = None
        self.stats = {
            'data_extraction': None,
            'cf_training': None,
            'survival_training': None,
            'total_duration': None,
            'success': False
        }

    def run_command(self, description, command, timeout=300):
        """Run a subprocess command with error handling"""
        logger.info(f"\n{'='*80}")
        logger.info(f"{description}")
        logger.info(f"{'='*80}")
        logger.info(f"Command: {' '.join(command)}")

        step_start = datetime.now()

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            duration = (datetime.now() - step_start).total_seconds()
            logger.info(f"✅ {description} completed in {duration:.1f} seconds")

            # Log output if verbose
            if result.stdout:
                logger.debug(f"STDOUT:\n{result.stdout}")

            return duration, True

        except subprocess.CalledProcessError as e:
            duration = (datetime.now() - step_start).total_seconds()
            logger.error(f"❌ {description} FAILED after {duration:.1f} seconds")
            logger.error(f"Exit code: {e.returncode}")
            logger.error(f"STDOUT:\n{e.stdout}")
            logger.error(f"STDERR:\n{e.stderr}")
            return duration, False

        except subprocess.TimeoutExpired as e:
            duration = timeout
            logger.error(f"❌ {description} TIMED OUT after {timeout} seconds")
            logger.error(f"STDOUT:\n{e.stdout}")
            logger.error(f"STDERR:\n{e.stderr}")
            return duration, False

        except Exception as e:
            duration = (datetime.now() - step_start).total_seconds()
            logger.error(f"❌ {description} FAILED with unexpected error: {e}")
            return duration, False

    def step1_extract_data(self):
        """Extract fresh data from MSSQL"""
        duration, success = self.run_command(
            "STEP 1: Extract fresh data from MSSQL",
            ['python3', 'scripts/generate_ml_features_direct.py'],
            timeout=120  # 2 minutes max
        )
        self.stats['data_extraction'] = duration
        return success

    def step2_train_collaborative_filtering(self):
        """Retrain ALS Collaborative Filtering model"""
        duration, success = self.run_command(
            "STEP 2: Train Collaborative Filtering (ALS)",
            ['python3', 'scripts/train_collaborative_filtering_v2.py'],
            timeout=120  # 2 minutes max
        )
        self.stats['cf_training'] = duration
        return success

    def step3_train_survival_analysis(self):
        """Retrain Survival Analysis model"""
        duration, success = self.run_command(
            "STEP 3: Train Survival Analysis (Weibull AFT)",
            ['python3', 'scripts/train_survival_analysis.py'],
            timeout=120  # 2 minutes max
        )
        self.stats['survival_training'] = duration
        return success

    def send_notification(self, success):
        """Send notification about refresh status (optional)"""
        # TODO: Configure email settings
        # For now, just log
        if success:
            logger.info("\n✅ DAILY REFRESH SUCCESSFUL - Models are up to date!")
        else:
            logger.error("\n❌ DAILY REFRESH FAILED - Manual intervention required!")

    def run(self):
        """Execute full daily refresh pipeline"""
        self.start_time = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("DAILY MODEL REFRESH PIPELINE - STARTING")
        logger.info("="*80)
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"Goal: Refresh data and retrain all ML models")

        try:
            # Step 1: Extract fresh data
            if not self.step1_extract_data():
                logger.error("Data extraction failed - aborting pipeline")
                self.stats['success'] = False
                self.send_notification(success=False)
                return False

            # Step 2: Train Collaborative Filtering
            if not self.step2_train_collaborative_filtering():
                logger.error("Collaborative Filtering training failed - aborting pipeline")
                self.stats['success'] = False
                self.send_notification(success=False)
                return False

            # Step 3: Train Survival Analysis
            if not self.step3_train_survival_analysis():
                logger.error("Survival Analysis training failed - aborting pipeline")
                self.stats['success'] = False
                self.send_notification(success=False)
                return False

            # Success!
            self.stats['success'] = True
            self.stats['total_duration'] = (datetime.now() - self.start_time).total_seconds()

            # Print summary
            self.print_summary()

            # Send notification
            self.send_notification(success=True)

            return True

        except Exception as e:
            logger.error(f"Pipeline failed with unexpected error: {e}", exc_info=True)
            self.stats['success'] = False
            self.send_notification(success=False)
            return False

    def print_summary(self):
        """Print summary statistics"""
        logger.info("\n" + "="*80)
        logger.info("DAILY REFRESH COMPLETE - SUMMARY")
        logger.info("="*80)

        logger.info(f"Start time: {self.start_time}")
        logger.info(f"End time: {datetime.now()}")
        logger.info(f"Total duration: {self.stats['total_duration']:.1f} seconds ({self.stats['total_duration']/60:.1f} minutes)")

        logger.info("\nStep-by-step timing:")
        logger.info(f"  1. Data extraction: {self.stats['data_extraction']:.1f}s")
        logger.info(f"  2. Collaborative Filtering: {self.stats['cf_training']:.1f}s")
        logger.info(f"  3. Survival Analysis: {self.stats['survival_training']:.1f}s")

        logger.info("\nModels updated:")
        logger.info("  ✅ models/collaborative_filtering/als_model_v2.pkl")
        logger.info("  ✅ models/survival_analysis/weibull_repurchase_model.pkl")

        logger.info("\nFresh data:")
        logger.info("  ✅ data/ml_features/concord_ml.duckdb")
        logger.info("  ✅ models/survival_analysis/reorder_alerts.csv (fresh HIGH priority alerts)")

        logger.info("\nNext steps:")
        logger.info("  → Reload models in API (if running)")
        logger.info("  → Export fresh alerts to CRM")
        logger.info("  → Next refresh: Tomorrow at 2 AM")

        logger.info("\n" + "="*80)


def main():
    """Main entry point"""
    pipeline = DailyRefreshPipeline()
    success = pipeline.run()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
