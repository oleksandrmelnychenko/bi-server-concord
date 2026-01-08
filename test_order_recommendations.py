#!/usr/bin/env python3
"""
Test script for /order-recommendations/v2 API endpoint.
Run: python test_order_recommendations.py
"""

import requests
import json
from datetime import datetime

API_URL = "http://localhost:8001"

def test_order_recommendations_v2():
    """Test the order recommendations v2 endpoint."""
    print("=" * 60)
    print("Testing /order-recommendations/v2 API")
    print("=" * 60)

    # Test request
    request_data = {
        "max_products": 10,
        "history_weeks": 26,
        "service_level": 0.95,
        "manufacturing_days": 14,
        "logistics_days": 21,
        "warehouse_days": 3,
        "use_trend_adjustment": True,
        "use_seasonality": True,
        "use_churn_adjustment": True
    }

    print(f"\nRequest: {json.dumps(request_data, indent=2)}")
    print("-" * 60)

    try:
        response = requests.post(
            f"{API_URL}/order-recommendations/v2",
            json=request_data,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        # Summary
        print(f"\n{'Metric':<25} {'Value':<30}")
        print("-" * 55)
        print(f"{'As of date':<25} {data.get('as_of_date', 'N/A'):<30}")
        print(f"{'Lead time days':<25} {data.get('lead_time_days', 'N/A'):<30}")
        print(f"{'Service level':<25} {data.get('service_level', 'N/A'):<30}")
        print(f"{'History weeks':<25} {data.get('history_weeks', 'N/A'):<30}")
        print(f"{'Products count':<25} {data.get('count', 0):<30}")
        print(f"{'Latency (ms)':<25} {data.get('latency_ms', 'N/A'):<30}")
        print(f"{'Products with trend':<25} {data.get('products_with_trend', 0):<30}")
        print(f"{'Products with season':<25} {data.get('products_with_seasonality', 0):<30}")
        print(f"{'Products with churn':<25} {data.get('products_with_churn_risk', 0):<30}")

        # Recommendations by supplier
        recommendations = data.get('recommendations', [])
        if recommendations:
            print(f"\n{'='*60}")
            print(f"Recommendations by Supplier ({len(recommendations)} suppliers)")
            print("=" * 60)

            total_qty = 0
            total_products = 0

            for rec in recommendations:
                supplier_name = rec.get('supplier_name', 'Unknown')[:40]
                products = rec.get('products', [])
                qty = rec.get('total_recommended_qty', 0)
                total_qty += qty
                total_products += len(products)

                print(f"\n{supplier_name}")
                print(f"  Products: {len(products)}, Total Qty: {qty}")

                # Show first 2 products
                for p in products[:2]:
                    name = (p.get('product_name') or '')[:35]
                    sku = p.get('vendor_code', '')
                    on_hand = p.get('on_hand', 0)
                    rec_qty = p.get('recommended_qty', 0)
                    confidence = p.get('forecast_confidence', 0) * 100

                    print(f"    - {name} ({sku})")
                    print(f"      On hand: {on_hand}, Recommended: {rec_qty}, Confidence: {confidence:.0f}%")

            print(f"\n{'='*60}")
            print(f"TOTALS: {total_products} products, {total_qty} units to order")
            print("=" * 60)

        # Validation checks
        print("\n" + "=" * 60)
        print("Validation Checks")
        print("=" * 60)

        checks = [
            ("Response has count", data.get('count') is not None),
            ("Response has recommendations", len(recommendations) > 0),
            ("Lead time calculated", data.get('lead_time_days') == 38),
            ("Latency reasonable (<5s)", data.get('latency_ms', 9999) < 5000),
            ("Has expected arrival date", recommendations and recommendations[0].get('products', [{}])[0].get('expected_arrival_date') is not None),
        ]

        all_passed = True
        for check_name, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check_name}")
            if not passed:
                all_passed = False

        print("\n" + ("ALL CHECKS PASSED!" if all_passed else "SOME CHECKS FAILED!"))
        return all_passed

    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to API at {API_URL}")
        print("Make sure the Main API is running on port 8001")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_health():
    """Test API health endpoint."""
    print("\nChecking API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        data = response.json()
        print(f"  Status: {data.get('status')}")
        print(f"  Version: {data.get('version')}")
        print(f"  Redis: {'connected' if data.get('redis_connected') else 'not connected'}")
        return data.get('status') == 'healthy'
    except Exception as e:
        print(f"  Health check failed: {e}")
        return False


if __name__ == "__main__":
    print(f"\nOrder Recommendations V2 API Test")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"API URL: {API_URL}")

    if test_health():
        success = test_order_recommendations_v2()
        exit(0 if success else 1)
    else:
        print("\nAPI is not healthy, skipping tests")
        exit(1)
