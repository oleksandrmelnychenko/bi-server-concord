#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test 30 complex queries against AI Provider API."""
import requests
import time
import json
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8001"

# 30 Complex Queries - Ukrainian
QUERIES = [
    # Category 1: Product Movement (5)
    "Покажи весь рух товару Continental за останні 5 місяців - закупки, залишки, продажі",
    "Які товари закупили у вересні але ще не продали",
    "Порівняй закупівельну і продажну ціну для топ 20 товарів",
    "Оборотність товарів по складах за жовтень",
    "Товари з від'ємним залишком та коли їх востаннє продавали",

    # Category 2: Financial Analytics (6)
    "Старіння дебіторки - скільки боргів 0-30 днів, 31-60, 61-90, більше 90",
    "Який був баланс клієнта АВТОТРАНС на 1 вересня 2025",
    "Виручка, платежі та борги по топ 20 клієнтах",
    "Топ 10 клієнтів з найстарішими боргами",
    "Динаміка платежів по місяцях за 2025 рік",
    "Маржинальність по категоріях товарів",

    # Category 3: Regional Analytics (4)
    "Топ 5 боржників по кожному регіону",
    "Продажі по регіонах за кожен квартал 2025",
    "Які регіони найбільше зросли у продажах порівняно з минулим роком",
    "Середній чек по регіонах",

    # Category 4: Client Analytics (5)
    "Виручка по типах клієнтів - ФОП, ТОВ, ПП окремо",
    "Клієнти які не замовляли більше 6 місяців",
    "Клієнти з найбільшою кількістю повернень товарів",
    "Загальна сума всіх замовлень кожного клієнта за весь час",
    "Нові клієнти за вересень 2025 з їх першим замовленням",

    # Category 5: Suppliers (4)
    "Топ 10 постачальників за сумою закупівель",
    "Середній час доставки від кожного постачальника",
    "Постачальники у яких є прострочені поставки",
    "Порівняй ціни на один товар від різних постачальників",

    # Category 6: Warehouse Analytics (3)
    "Залишки товарів по всіх складах з загальною вартістю",
    "Що є на складі браку і на яку суму",
    "Які склади мали найбільший оборот у вересні",

    # Category 7: Time Comparisons (3)
    "Порівняй продажі вересня 2024 і вересня 2025",
    "Тренд продажів по тижнях за 3 квартал 2025",
    "Сезонність - продажі по місяцях за 2024 і 2025 роки",
]

def test_query(question, timeout=180):
    """Test a single query and return detailed results."""
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": question},
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        elapsed = time.time() - start
        data = response.json()

        return {
            "question": question,
            "success": data.get('success', False),
            "sql": data.get('sql', ''),
            "error": data.get('error', ''),
            "row_count": data.get('row_count', 0),
            "time": elapsed,
        }
    except Exception as e:
        return {
            "question": question,
            "success": False,
            "sql": "",
            "error": str(e),
            "row_count": 0,
            "time": 0,
        }

def run_all_tests():
    """Run all 30 tests and return results."""
    print("=" * 80)
    print(f"TESTING 30 COMPLEX QUERIES - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    results = []
    success_count = 0

    for i, query in enumerate(QUERIES, 1):
        print(f"[{i:02d}/30] Testing: {query[:60]}...")
        result = test_query(query)
        results.append(result)

        if result['success']:
            success_count += 1
            print(f"        OK ({result['time']:.1f}s) - {result['row_count']} rows")
        else:
            error_short = result['error'][:80] if result['error'] else 'Unknown'
            print(f"        FAIL ({result['time']:.1f}s) - {error_short}")

        # Small delay to not overwhelm the API
        time.sleep(0.5)

    print()
    print("=" * 80)
    print(f"RESULTS: {success_count}/30 ({100*success_count//30}%) SUCCESS")
    print("=" * 80)

    # Print failed queries
    failed = [r for r in results if not r['success']]
    if failed:
        print(f"\nFAILED QUERIES ({len(failed)}):")
        print("-" * 80)
        for i, r in enumerate(failed, 1):
            print(f"\n{i}. {r['question']}")
            print(f"   Error: {r['error'][:200]}")
            if r['sql']:
                print(f"   SQL: {r['sql'][:200]}...")

    # Print successful queries with SQL
    successful = [r for r in results if r['success']]
    if successful:
        print(f"\n\nSUCCESSFUL QUERIES ({len(successful)}):")
        print("-" * 80)
        for i, r in enumerate(successful, 1):
            print(f"\n{i}. {r['question']}")
            print(f"   Rows: {r['row_count']}, Time: {r['time']:.1f}s")
            print(f"   SQL: {r['sql'][:300]}...")

    # Save results to file
    with open('test_30_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'success_count': success_count,
            'total': 30,
            'success_rate': f"{100*success_count//30}%",
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n\nResults saved to test_30_results.json")

    return results, success_count

if __name__ == "__main__":
    results, success = run_all_tests()
