"""Test the API endpoints."""
import requests
import json

BASE_URL = "http://localhost:8003"

def test_api():
    print("=" * 70)
    print("TESTING DB AI API")
    print("=" * 70)

    # Test root
    print("\n1. Testing root endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"   Status: {r.status_code}")
        print(f"   Response: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    # Test health
    print("\n2. Testing health endpoint...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"   Status: {r.status_code}")
        print(f"   Response: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test Ukrainian query
    print("\n3. Testing Ukrainian query...")
    try:
        r = requests.post(
            f"{BASE_URL}/query",
            json={
                "question": "Покажи топ 10 товарів за продажами",
                "execute": False
            },
            timeout=120
        )
        print(f"   Status: {r.status_code}")
        data = r.json()
        if "sql" in data:
            print(f"   SQL: {data['sql']}")
        else:
            print(f"   Response: {data}")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test English query
    print("\n4. Testing English query...")
    try:
        r = requests.post(
            f"{BASE_URL}/query",
            json={
                "question": "Show total revenue this year",
                "execute": False
            },
            timeout=120
        )
        print(f"   Status: {r.status_code}")
        data = r.json()
        if "sql" in data:
            print(f"   SQL: {data['sql']}")
        else:
            print(f"   Response: {data}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "=" * 70)
    print("API TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_api()
