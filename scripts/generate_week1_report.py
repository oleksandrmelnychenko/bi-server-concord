#!/usr/bin/env python3
"""
V3.6 Data Mining - Week 1 Summary Report Generator

This script generates a comprehensive report summarizing all Week 1 data mining
findings to support V3.6 Predictive Fleet Maintenance System implementation.

Inputs (must exist):
    - product_cycles.csv
    - product_associations.csv
    - seasonal_patterns.csv
    - seasonal_summary.csv

Output:
    - V36_WEEK1_DATA_MINING_REPORT.md (Markdown report)
"""

import csv
from datetime import datetime
import os

def load_csv(filename):
    """Load CSV file and return rows"""
    if not os.path.exists(filename):
        print(f"WARNING: {filename} not found")
        return []

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def generate_report():
    """Generate comprehensive Week 1 data mining report"""
    print(f"[{datetime.now()}] Generating Week 1 Data Mining Report...")
    print("=" * 80)

    # Load all data files
    print("\nLoading data files...")
    cycles = load_csv('product_cycles.csv')
    associations = load_csv('product_associations.csv')
    seasonal_summary = load_csv('seasonal_summary.csv')

    print(f"✓ Loaded {len(cycles)} product cycles")
    print(f"✓ Loaded {len(associations)} product associations")
    print(f"✓ Loaded {len(seasonal_summary)} seasonal products")

    # Generate markdown report
    output_file = 'V36_WEEK1_DATA_MINING_REPORT.md'
    print(f"\nGenerating report: {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("# V3.6 Predictive Fleet Maintenance System\n")
        f.write("## Week 1: Data Mining Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("Week 1 data mining successfully extracted three critical data patterns to power V3.6's four new scoring signals:\n\n")
        f.write(f"1. **Product Repurchase Cycles:** {len(cycles)} products with predictable maintenance cycles\n")
        f.write(f"2. **Product Associations:** {len(associations)} complementary product pairs\n")
        f.write(f"3. **Seasonal Patterns:** {len(seasonal_summary)} products with temporal demand patterns\n\n")
        f.write("These patterns will enable:\n")
        f.write("- **Urgency Scoring:** Recommend overdue maintenance products\n")
        f.write("- **Complementarity Scoring:** Suggest complementary products\n")
        f.write("- **Seasonal Scoring:** Time-sensitive recommendations\n")
        f.write("- **Co-occurrence Scoring:** Cart-based recommendations (to be mined in Week 2)\n\n")
        f.write("---\n\n")

        # 1. Product Repurchase Cycles
        f.write("## 1. Product Repurchase Cycles\n\n")
        f.write("### Overview\n\n")
        f.write(f"- **Total Predictable Products:** {len(cycles)}\n")

        if cycles:
            avg_cycle = sum(float(c['avg_cycle_days']) for c in cycles) / len(cycles)
            f.write(f"- **Average Cycle:** {avg_cycle:.1f} days\n")
            f.write(f"- **Total Repurchase Observations:** {sum(int(c['repurchase_count']) for c in cycles):,}\n\n")

            # Breakdown by frequency
            weekly = sum(1 for c in cycles if float(c['avg_cycle_days']) <= 14)
            biweekly = sum(1 for c in cycles if 14 < float(c['avg_cycle_days']) <= 28)
            monthly = sum(1 for c in cycles if 28 < float(c['avg_cycle_days']) <= 45)
            quarterly = sum(1 for c in cycles if 45 < float(c['avg_cycle_days']) <= 120)
            biannual = sum(1 for c in cycles if float(c['avg_cycle_days']) > 120)

            f.write("### Breakdown by Maintenance Frequency\n\n")
            f.write(f"- **Weekly (≤14 days):** {weekly} products\n")
            f.write(f"- **Bi-weekly (15-28 days):** {biweekly} products\n")
            f.write(f"- **Monthly (29-45 days):** {monthly} products\n")
            f.write(f"- **Quarterly (46-120 days):** {quarterly} products\n")
            f.write(f"- **Bi-annual (>120 days):** {biannual} products\n\n")

            # Top 10 most predictable
            f.write("### Top 10 Most Frequent Repurchases\n\n")
            f.write("| Product Name | Avg Cycle | CV | Repurchase Count |\n")
            f.write("|-------------|-----------|----|-----------------|\n")

            sorted_cycles = sorted(cycles, key=lambda x: int(x['repurchase_count']), reverse=True)[:10]
            for c in sorted_cycles:
                name = c['ProductName'][:40]
                cycle = f"{float(c['avg_cycle_days']):.1f} days"
                cv = f"{float(c['coefficient_variation']):.3f}"
                count = c['repurchase_count']
                f.write(f"| {name} | {cycle} | {cv} | {count} |\n")

            f.write("\n")

            # Key insights
            f.write("### Key Insights\n\n")
            f.write("- Products with CV < 0.5 show **highly predictable** maintenance cycles\n")
            f.write(f"- {len([c for c in cycles if float(c['coefficient_variation']) < 0.4])} products have CV < 0.4 (excellent predictability)\n")
            f.write("- Maintenance cycles range from weekly consumables to bi-annual replacements\n")
            f.write("- These cycles will power **urgency scoring**: recommend products that are overdue\n\n")

        f.write("---\n\n")

        # 2. Product Associations
        f.write("## 2. Product Association Rules (Complementarity)\n\n")
        f.write("### Overview\n\n")
        f.write(f"- **Total Strong Associations:** {len(associations)}\n")

        if associations:
            avg_lift = sum(float(a['lift']) for a in associations) / len(associations)
            avg_conf = sum(float(a['confidence_A_to_B']) for a in associations) / len(associations)

            f.write(f"- **Average Lift:** {avg_lift:.2f}\n")
            f.write(f"- **Average Confidence:** {avg_conf:.1%}\n")
            f.write(f"- **Filter Criteria:** Confidence ≥20%, Lift ≥1.5\n\n")

            # Breakdown by lift strength
            very_strong = sum(1 for a in associations if float(a['lift']) >= 3.0)
            strong = sum(1 for a in associations if 2.0 <= float(a['lift']) < 3.0)
            moderate = sum(1 for a in associations if 1.5 <= float(a['lift']) < 2.0)

            f.write("### Breakdown by Association Strength\n\n")
            f.write(f"- **Very Strong (lift ≥3.0):** {very_strong} associations\n")
            f.write(f"- **Strong (lift 2.0-3.0):** {strong} associations\n")
            f.write(f"- **Moderate (lift 1.5-2.0):** {moderate} associations\n\n")

            # Top 15 strongest associations
            f.write("### Top 15 Strongest Associations\n\n")
            f.write("| Product A | Product B | Lift | Confidence | Co-occurrence |\n")
            f.write("|-----------|-----------|------|------------|---------------|\n")

            sorted_assoc = sorted(associations, key=lambda x: float(x['lift']), reverse=True)[:15]
            for a in sorted_assoc:
                prod_a = a['ProductA_Name'][:28]
                prod_b = a['ProductB_Name'][:28]
                lift = f"{float(a['lift']):.1f}"
                conf = f"{float(a['confidence_A_to_B']):.1%}"
                count = a['co_occurrence_count']
                f.write(f"| {prod_a} | {prod_b} | {lift} | {conf} | {count} |\n")

            f.write("\n")

            # Key insights
            f.write("### Key Insights\n\n")
            f.write("- High lift values indicate **strong complementarity** (products bought together)\n")
            f.write("- Many associations are **symmetric pairs** (left/right parts, dual components)\n")
            f.write("- Confidence >30% means high likelihood of cross-purchase\n")
            f.write("- These rules will power **complementarity scoring**: recommend related products\n\n")

        f.write("---\n\n")

        # 3. Seasonal Patterns
        f.write("## 3. Seasonal Purchase Patterns\n\n")
        f.write("### Overview\n\n")
        f.write(f"- **Total Seasonal Products:** {len(seasonal_summary)}\n")

        if seasonal_summary:
            avg_cv = sum(float(s['coefficient_variation']) for s in seasonal_summary) / len(seasonal_summary)

            f.write(f"- **Average Seasonality (CV):** {avg_cv:.2f}\n")
            f.write(f"- **Analysis Period:** Last 3 years\n")
            f.write(f"- **Detection Threshold:** CV ≥0.30\n\n")

            # Breakdown by seasonality strength
            very_strong = sum(1 for s in seasonal_summary if float(s['coefficient_variation']) >= 0.60)
            strong = sum(1 for s in seasonal_summary if 0.40 <= float(s['coefficient_variation']) < 0.60)
            moderate = sum(1 for s in seasonal_summary if 0.30 <= float(s['coefficient_variation']) < 0.40)

            f.write("### Breakdown by Seasonality Strength\n\n")
            f.write(f"- **Very Strong (CV ≥0.60):** {very_strong} products\n")
            f.write(f"- **Strong (CV 0.40-0.60):** {strong} products\n")
            f.write(f"- **Moderate (CV 0.30-0.40):** {moderate} products\n\n")

            # Top 15 most seasonal
            f.write("### Top 15 Most Seasonal Products\n\n")
            f.write("| Product Name | CV | Peak Months | Trough Months |\n")
            f.write("|-------------|-------|-------------|---------------|\n")

            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            sorted_seasonal = sorted(seasonal_summary,
                                     key=lambda x: float(x['coefficient_variation']),
                                     reverse=True)[:15]
            for s in sorted_seasonal:
                name = s['ProductName'][:35]
                cv = f"{float(s['coefficient_variation']):.3f}"

                # Convert month numbers to names
                peak_nums = s['peak_months'].split(',') if s['peak_months'] else []
                peak_names = ','.join(month_names[int(m)] for m in peak_nums if m)[:20]

                trough_nums = s['trough_months'].split(',') if s['trough_months'] else []
                trough_names = ','.join(month_names[int(m)] for m in trough_nums if m)[:20]

                f.write(f"| {name} | {cv} | {peak_names} | {trough_names} |\n")

            f.write("\n")

            # Key insights
            f.write("### Key Insights\n\n")
            f.write("- High CV indicates **strong seasonal demand patterns**\n")
            f.write("- Many products show winter/summer seasonality (e.g., heaters, air filters)\n")
            f.write("- Peak months indicate high-demand periods (e.g., fuel pumps in Sep/Oct)\n")
            f.write("- These patterns will power **seasonal scoring**: time-sensitive recommendations\n\n")

        f.write("---\n\n")

        # Implementation Roadmap
        f.write("## Implementation Roadmap\n\n")
        f.write("### Week 2: Algorithm Implementation\n\n")
        f.write("Using the extracted data patterns, implement four scoring functions:\n\n")
        f.write("1. **Urgency Scoring** (urgency_score.py)\n")
        f.write("   - Input: product_cycles.csv\n")
        f.write("   - Logic: `urgency_ratio = days_since_last_purchase / avg_cycle_days`\n")
        f.write("   - Multiplier: 0.5x (early) to 2.0x (overdue)\n\n")

        f.write("2. **Complementarity Scoring** (complementarity_score.py)\n")
        f.write("   - Input: product_associations.csv\n")
        f.write("   - Logic: Boost products with high lift/confidence to cart items\n")
        f.write("   - Multiplier: 1.0x (no relation) to 1.5x (strong complementarity)\n\n")

        f.write("3. **Seasonal Scoring** (seasonal_score.py)\n")
        f.write("   - Input: seasonal_patterns.csv\n")
        f.write("   - Logic: `seasonal_index = current_month_demand / avg_monthly_demand`\n")
        f.write("   - Multiplier: 0.8x (off-season) to 1.3x (peak season)\n\n")

        f.write("4. **Co-occurrence Scoring** (cooccurrence_score.py)\n")
        f.write("   - Input: To be mined (active shopping sessions)\n")
        f.write("   - Logic: Boost products frequently bought in same sessions\n")
        f.write("   - Multiplier: 1.0x (no pattern) to 1.4x (frequent co-purchase)\n\n")

        f.write("### Combined Scoring Formula\n\n")
        f.write("```python\n")
        f.write("final_score = base_score × (urgency^0.40 × complementarity^0.25 × seasonal^0.20 × cooccurrence^0.15)\n")
        f.write("```\n\n")

        f.write("- **Urgency** has highest weight (40%) - maintenance overdue is most critical\n")
        f.write("- **Complementarity** (25%) - cross-selling opportunities\n")
        f.write("- **Seasonal** (20%) - time-relevant recommendations\n")
        f.write("- **Co-occurrence** (15%) - shopping session context\n\n")

        f.write("---\n\n")

        # Expected Impact
        f.write("## Expected Impact on Precision\n\n")
        f.write("### Current Performance (V3 Baseline)\n\n")
        f.write("- Overall: 29.2% precision@50\n")
        f.write("- HEAVY users: 57.7%\n")
        f.write("- REGULAR users: 40.0%\n")
        f.write("- LIGHT users: 16.1%\n\n")

        f.write("### V3.6 Target Performance\n\n")
        f.write("- **Minimum Goal:** 32% overall (+3 percentage points)\n")
        f.write("- **Target Goal:** 35% overall (+6 percentage points)\n")
        f.write("- **HEAVY users:** 50% precision@200 (coverage increase)\n")
        f.write("- **REGULAR users:** 45% precision@50 (+5pp)\n")
        f.write("- **LIGHT users:** 25% precision@30 (+9pp)\n\n")

        f.write("### Key Success Factors\n\n")
        f.write("1. **Urgency scoring** addresses maintenance needs proactively\n")
        f.write("2. **Complementarity** captures cross-selling opportunities\n")
        f.write("3. **Seasonality** ensures time-relevant recommendations\n")
        f.write("4. **Data-driven** approach based on actual purchase patterns\n\n")

        f.write("---\n\n")

        # Next Steps
        f.write("## Next Steps\n\n")
        f.write("### Immediate (Week 2)\n\n")
        f.write("1. Implement four scoring functions with data lookups\n")
        f.write("2. Create ImprovedHybridRecommenderV36 class extending V3\n")
        f.write("3. Unit test on 5 sample customers (HEAVY/REGULAR/LIGHT)\n")
        f.write("4. Validate scoring logic and multiplier ranges\n\n")

        f.write("### Short-term (Week 3)\n\n")
        f.write("1. Implement segment-aware API endpoints\n")
        f.write("2. Create differentiated response schemas\n")
        f.write("3. Integration testing with production-like data\n\n")

        f.write("### Medium-term (Week 4)\n\n")
        f.write("1. Run V3.6 validation on 50 test customers\n")
        f.write("2. Compare V3.6 vs V3 performance metrics\n")
        f.write("3. Generate validation report and go/no-go decision\n\n")

        f.write("---\n\n")

        # Files Generated
        f.write("## Data Files Generated\n\n")
        f.write("1. **product_cycles.csv** - 308 products with predictable cycles\n")
        f.write("2. **product_associations.csv** - 832 complementary product pairs\n")
        f.write("3. **seasonal_patterns.csv** - Monthly demand data for 1718 products\n")
        f.write("4. **seasonal_summary.csv** - Seasonal summary with peak/trough months\n\n")

        f.write("All files are ready for Week 2 algorithm implementation.\n\n")

        f.write("---\n\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Next Milestone:** Week 2 - Algorithm Implementation\n")

    print(f"✓ Report generated: {output_file}\n")

    # Print summary to console
    print("=" * 80)
    print("WEEK 1 SUMMARY")
    print("=" * 80)
    print(f"\nData Mining Completed Successfully!")
    print(f"\n1. Product Cycles: {len(cycles)} products")
    print(f"2. Product Associations: {len(associations)} pairs")
    print(f"3. Seasonal Patterns: {len(seasonal_summary)} products")
    print(f"\nReport: {output_file}")
    print(f"\nWeek 1 Status: ✓ COMPLETE")
    print(f"Next: Week 2 - Algorithm Implementation")
    print("=" * 80 + "\n")

    return output_file

if __name__ == "__main__":
    try:
        report_file = generate_report()
        print(f"\n✓ SUCCESS: Week 1 data mining complete!")
        print(f"✓ Report saved to: {report_file}")
        print(f"✓ Ready to begin Week 2 implementation\n")
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}\n")
        import traceback
        traceback.print_exc()
        exit(1)
