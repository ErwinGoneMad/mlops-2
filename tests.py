import requests
import time

API_BASE_URL = "http://localhost:8000"


def test_predict_endpoint():
    print("\nTest du endpoint /predict...")

    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
        )

        print(f"   Status: {response.status_code}")
        print(f"   Request: {test_data}")
        print(f"   Response: {response.json()}")

        return response.status_code == 200
    except Exception as e:
        print(f"   Erreur: {e}")
        return False


def test_update_model_endpoint():
    print("\nTest du endpoint /update-model...")

    update_data = {"version": "latest"}

    try:
        response = requests.post(
            f"{API_BASE_URL}/update-model",
            json=update_data,
            headers={"Content-Type": "application/json"},
        )

        print(f"   Status: {response.status_code}")
        print(f"   Request: {update_data}")
        print(f"   Response: {response.json()}")

        return response.status_code == 200
    except Exception as e:
        print(f"   Erreur: {e}")
        return False


def main():
    time.sleep(2)

    tests = [
        test_predict_endpoint,
        test_update_model_endpoint,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if test_func():
            passed += 1

    print(f"Résultats: {passed}/{total} tests réussis")

    if passed == total:
        print("Tous les tests sont passés !")
    else:
        print("Certains tests ont échoué.")


if __name__ == "__main__":
    main()
