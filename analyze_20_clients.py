#!/usr/bin/env python3
"""
Analyze API Recommendations for 20 Sample Clients

Tests the V3.2 recommendation API on 20 diverse clients and performs:
1. Quality analysis (discovery rate, diversity, segment distribution)
2. Performance analysis (latency, throughput)
3. Business value analysis (repurchase vs discovery mix)
4. Product analysis (overlap, uniqueness, coverage)

Usage:
    python3 analyze_20_clients.py
"""

import requests
import json
import time
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Any
import statistics

# Configuration
API_URL = "http://localhost:8000"
AS_OF_DATE = "2024-07-01"
TOP_N = 50
INCLUDE_DISCOVERY = True


def get_sample_clients(conn) -> List[int]:
    """Get 20 diverse sample clients (mix of Heavy, Regular, Light)"""
    cursor = conn.cursor()

    # Get mix of segments
    query = """
    SELECT TOP 20
        ca.ClientID,
        COUNT(DISTINCT o.ID) as order_count,
        COUNT(DISTINCT oi.ProductID) as product_count
    FROM dbo.ClientAgreement ca
    INNER JOIN dbo.[Order] o ON ca.ID = o.ClientAgreementID
    INNER JOIN dbo.OrderItem oi ON o.ID = oi.OrderID
    WHERE o.Created >= DATEADD(day, -365, '2024-07-01')
          AND o.Created < '2024-07-01'
    GROUP BY ca.ClientID
    HAVING COUNT(DISTINCT o.ID) >= 5  -- Active customers
    ORDER BY NEWID()  -- Random sample
    """

    cursor.execute(query)
    clients = [row[0] for row in cursor.fetchall()]
    cursor.close()

    return clients


def call_api(customer_id: int) -> Dict[str, Any]:
    """Call recommendation API for a customer"""
    url = f"{API_URL}/recommend"
    payload = {
        "customer_id": customer_id,
        "as_of_date": AS_OF_DATE,
        "top_n": TOP_N,
        "include_discovery": INCLUDE_DISCOVERY,
        "use_cache": False
    }

    start_time = time.time()
    response = requests.post(url, json=payload)
    latency = (time.time() - start_time) * 1000  # Convert to ms

    if response.status_code == 200:
        data = response.json()
        data['api_latency_ms'] = latency
        return data
    else:
        raise Exception(f"API error: {response.status_code} - {response.text}")


def analyze_recommendations(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze recommendation results"""

    analysis = {
        'total_clients': len(results),
        'total_recommendations': 0,
        'avg_recommendations_per_client': 0,
        'segments': Counter(),
        'sources': Counter(),
        'discovery_stats': {
            'clients_with_discovery': 0,
            'total_discovery': 0,
            'avg_discovery_per_client': 0,
            'discovery_rate': 0.0
        },
        'performance': {
            'avg_latency_ms': 0,
            'min_latency_ms': float('inf'),
            'max_latency_ms': 0,
            'p50_latency_ms': 0,
            'p95_latency_ms': 0,
            'p99_latency_ms': 0
        },
        'diversity': {
            'unique_products': set(),
            'product_frequency': Counter(),
            'overlap_matrix': defaultdict(lambda: defaultdict(int))
        },
        'by_segment': defaultdict(lambda: {
            'count': 0,
            'total_recs': 0,
            'discovery_count': 0,
            'avg_latency_ms': 0
        })
    }

    latencies = []
    all_products = []

    for result in results:
        recs = result['recommendations']
        segment = recs[0]['segment'] if recs else 'UNKNOWN'
        latency = result.get('api_latency_ms', 0)

        # Overall stats
        analysis['total_recommendations'] += len(recs)
        analysis['segments'][segment] += 1
        latencies.append(latency)

        # Source breakdown
        discovery_count = 0
        for rec in recs:
            analysis['sources'][rec['source']] += 1
            analysis['diversity']['unique_products'].add(rec['product_id'])
            analysis['diversity']['product_frequency'][rec['product_id']] += 1
            all_products.append(rec['product_id'])

            if rec['source'] in ['discovery', 'hybrid']:
                discovery_count += 1

        # Discovery stats
        if discovery_count > 0:
            analysis['discovery_stats']['clients_with_discovery'] += 1
        analysis['discovery_stats']['total_discovery'] += discovery_count

        # By segment
        seg_stats = analysis['by_segment'][segment]
        seg_stats['count'] += 1
        seg_stats['total_recs'] += len(recs)
        seg_stats['discovery_count'] += discovery_count
        seg_stats['avg_latency_ms'] += latency

    # Calculate averages
    analysis['avg_recommendations_per_client'] = analysis['total_recommendations'] / len(results)
    analysis['discovery_stats']['avg_discovery_per_client'] = analysis['discovery_stats']['total_discovery'] / len(results)
    analysis['discovery_stats']['discovery_rate'] = analysis['discovery_stats']['total_discovery'] / analysis['total_recommendations']

    # Performance percentiles
    latencies.sort()
    analysis['performance']['avg_latency_ms'] = statistics.mean(latencies)
    analysis['performance']['min_latency_ms'] = min(latencies)
    analysis['performance']['max_latency_ms'] = max(latencies)
    analysis['performance']['p50_latency_ms'] = statistics.median(latencies)
    analysis['performance']['p95_latency_ms'] = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
    analysis['performance']['p99_latency_ms'] = latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0]

    # Segment averages
    for segment, stats in analysis['by_segment'].items():
        if stats['count'] > 0:
            stats['avg_recs'] = stats['total_recs'] / stats['count']
            stats['avg_discovery'] = stats['discovery_count'] / stats['count']
            stats['avg_latency_ms'] = stats['avg_latency_ms'] / stats['count']

    # Diversity metrics
    analysis['diversity']['total_unique_products'] = len(analysis['diversity']['unique_products'])
    analysis['diversity']['avg_product_frequency'] = statistics.mean(analysis['diversity']['product_frequency'].values())
    analysis['diversity']['product_overlap_rate'] = len([p for p, count in analysis['diversity']['product_frequency'].items() if count > 1]) / len(analysis['diversity']['unique_products'])

    return analysis


def print_analysis(analysis: Dict[str, Any], results: List[Dict[str, Any]]):
    """Print comprehensive analysis report"""

    print("=" * 80)
    print("V3.2 API ANALYSIS - 20 CLIENT SAMPLE")
    print("=" * 80)
    print(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API: {API_URL}")
    print(f"As-of Date: {AS_OF_DATE}")
    print(f"Top N: {TOP_N}")

    # 1. OVERALL STATISTICS
    print("\n" + "=" * 80)
    print("1. OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total Clients Tested: {analysis['total_clients']}")
    print(f"Total Recommendations: {analysis['total_recommendations']}")
    print(f"Avg Recommendations/Client: {analysis['avg_recommendations_per_client']:.1f}")

    # 2. SEGMENT DISTRIBUTION
    print("\n" + "=" * 80)
    print("2. SEGMENT DISTRIBUTION")
    print("=" * 80)
    print(f"{'Segment':<25} {'Count':<10} {'Percentage':<12}")
    print("-" * 50)
    for segment, count in analysis['segments'].most_common():
        pct = count / analysis['total_clients'] * 100
        print(f"{segment:<25} {count:<10} {pct:<12.1f}%")

    # 3. RECOMMENDATION SOURCES
    print("\n" + "=" * 80)
    print("3. RECOMMENDATION SOURCES (Repurchase vs Discovery)")
    print("=" * 80)
    print(f"{'Source':<25} {'Count':<10} {'Percentage':<12}")
    print("-" * 50)
    for source, count in analysis['sources'].most_common():
        pct = count / analysis['total_recommendations'] * 100
        print(f"{source:<25} {count:<10} {pct:<12.1f}%")

    # 4. DISCOVERY ANALYSIS
    print("\n" + "=" * 80)
    print("4. DISCOVERY ANALYSIS")
    print("=" * 80)
    print(f"Clients with Discovery: {analysis['discovery_stats']['clients_with_discovery']}/{analysis['total_clients']} " +
          f"({analysis['discovery_stats']['clients_with_discovery']/analysis['total_clients']*100:.1f}%)")
    print(f"Total Discovery Products: {analysis['discovery_stats']['total_discovery']}")
    print(f"Avg Discovery/Client: {analysis['discovery_stats']['avg_discovery_per_client']:.2f}")
    print(f"Discovery Rate: {analysis['discovery_stats']['discovery_rate']*100:.1f}%")

    # 5. PERFORMANCE METRICS
    print("\n" + "=" * 80)
    print("5. PERFORMANCE METRICS")
    print("=" * 80)
    perf = analysis['performance']
    print(f"Average Latency: {perf['avg_latency_ms']:.0f}ms")
    print(f"Min Latency: {perf['min_latency_ms']:.0f}ms")
    print(f"Max Latency: {perf['max_latency_ms']:.0f}ms")
    print(f"P50 Latency: {perf['p50_latency_ms']:.0f}ms")
    print(f"P95 Latency: {perf['p95_latency_ms']:.0f}ms")
    print(f"P99 Latency: {perf['p99_latency_ms']:.0f}ms")

    # 6. BY SEGMENT BREAKDOWN
    print("\n" + "=" * 80)
    print("6. ANALYSIS BY SEGMENT")
    print("=" * 80)
    print(f"{'Segment':<25} {'Clients':<10} {'Avg Recs':<12} {'Avg Discovery':<15} {'Avg Latency':<15}")
    print("-" * 80)
    for segment, stats in analysis['by_segment'].items():
        print(f"{segment:<25} {stats['count']:<10} {stats['avg_recs']:<12.1f} " +
              f"{stats['avg_discovery']:<15.1f} {stats['avg_latency_ms']:<15.0f}ms")

    # 7. DIVERSITY ANALYSIS
    print("\n" + "=" * 80)
    print("7. PRODUCT DIVERSITY ANALYSIS")
    print("=" * 80)
    div = analysis['diversity']
    print(f"Total Unique Products Recommended: {div['total_unique_products']}")
    print(f"Total Recommendations: {analysis['total_recommendations']}")
    print(f"Avg Product Frequency: {div['avg_product_frequency']:.2f} (how many times each product appears)")
    print(f"Product Overlap Rate: {div['product_overlap_rate']*100:.1f}% (products recommended to multiple clients)")

    # Top 10 most frequently recommended products
    print(f"\nTop 10 Most Frequently Recommended Products:")
    print(f"{'Product ID':<15} {'Frequency':<12} {'Percentage':<12}")
    print("-" * 40)
    for product_id, count in div['product_frequency'].most_common(10):
        pct = count / analysis['total_clients'] * 100
        print(f"{product_id:<15} {count:<12} {pct:<12.1f}%")

    # 8. SAMPLE RECOMMENDATIONS
    print("\n" + "=" * 80)
    print("8. SAMPLE RECOMMENDATIONS (First 5 Clients)")
    print("=" * 80)
    for i, result in enumerate(results[:5], 1):
        recs = result['recommendations']
        if recs:
            segment = recs[0]['segment']
            discovery_count = sum(1 for r in recs if r['source'] in ['discovery', 'hybrid'])
            print(f"\nClient #{i}: {result['customer_id']} ({segment})")
            print(f"  Total: {len(recs)} recs, Discovery: {discovery_count}, Latency: {result.get('api_latency_ms', 0):.0f}ms")
            print(f"  Top 5 products:")
            for rec in recs[:5]:
                print(f"    #{rec['rank']}: Product {rec['product_id']} (score: {rec['score']:.4f}, {rec['source']})")


def save_results(results: List[Dict[str, Any]], analysis: Dict[str, Any]):
    """Save results to JSON files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save raw results
    results_file = f"analysis_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Raw results saved to: {results_file}")

    # Save analysis
    analysis_file = f"analysis_report_{timestamp}.json"
    # Convert sets to lists for JSON serialization
    analysis['diversity']['unique_products'] = list(analysis['diversity']['unique_products'])
    analysis['diversity']['product_frequency'] = dict(analysis['diversity']['product_frequency'])

    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"✅ Analysis report saved to: {analysis_file}")


def main():
    """Main execution"""
    print("Starting 20-client API analysis...")
    print("=" * 80)

    # Get sample clients from database
    from api.db_pool import get_connection
    conn = get_connection()

    print("\n1. Getting sample clients from database...")
    clients = get_sample_clients(conn)
    print(f"   ✅ Found {len(clients)} clients: {clients[:5]}... (showing first 5)")

    conn.close()

    # Call API for each client
    print("\n2. Calling API for each client...")
    results = []
    for i, customer_id in enumerate(clients, 1):
        try:
            print(f"   [{i}/{len(clients)}] Calling API for customer {customer_id}...", end=" ")
            result = call_api(customer_id)
            results.append(result)

            recs = result['recommendations']
            discovery_count = sum(1 for r in recs if r['source'] in ['discovery', 'hybrid'])
            print(f"✅ {len(recs)} recs ({discovery_count} discovery), {result['api_latency_ms']:.0f}ms")

        except Exception as e:
            print(f"❌ Error: {e}")

    if not results:
        print("\n❌ No results collected. Exiting.")
        return

    # Analyze results
    print("\n3. Analyzing results...")
    analysis = analyze_recommendations(results)
    print("   ✅ Analysis complete")

    # Print report
    print("\n4. Generating report...")
    print_analysis(analysis, results)

    # Save results
    print("\n5. Saving results...")
    save_results(results, analysis)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
