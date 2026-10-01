# simulations/run_simulation.py
import time
import argparse
import numpy as np
import requests

FEATURE_STATS = {
    "duration": (20.9, 12.0, 4.0, 72.0),
    "credit_amount": (3271.0, 2822.0, 250.0, 50000.0),
    "installment_commitment": (2.97, 1.11, 1.0, 4.0),
    "residence_since": (2.84, 1.10, 1.0, 4.0),
    "age": (35.5, 11.3, 19.0, 75.0),
    "existing_credits": (1.40, 0.57, 1.0, 4.0),
    "num_dependents": (1.15, 0.36, 1.0, 2.0),
    "is_male": (0.69, 0.46, 0.0, 1.0),
}


def generate_sample(drift_magnitude: float = 0.0):
    features_list = []
    features_dict = {}
    for name, (mean, std, min_v, max_v) in FEATURE_STATS.items():
        if name == "is_male":
            val = float(np.random.choice([0.0, 1.0], p=[0.31, 0.69]))
        else:
            shifted_mean = mean + (drift_magnitude * std)
            val = float(np.clip(np.random.normal(shifted_mean, std), min_v, max_v))
            if name in ["duration", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"]:
                val = float(round(val))
        features_list.append(val)
        features_dict[name] = val
    return features_list, features_dict


def run(scenario: str, iterations: int, api_url: str, evidently_url: str):
    print(f"Bắt đầu chạy kịch bản '{scenario}' ({iterations} requests)...")
    for i in range(iterations):
        if scenario == "normal":
            drift = 0.0
        elif scenario == "gradual_drift":
            drift = (i / iterations) * 2.5
        elif scenario == "sudden_shift":
            drift = 3.0 if i > (iterations // 3) else 0.0
        else:
            drift = 0.0

        f_list, f_dict = generate_sample(drift_magnitude=drift)
        try:
            resp = requests.post(f"{api_url}/predict", json={"features": f_list}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                requests.post(
                    f"{evidently_url}/iterate",
                    json={
                        "features": f_dict,
                        "prediction": data["prediction"],
                        "confidence": data["confidence"]
                    },
                    timeout=5
                )
        except Exception as e:
            print(f"Request {i} lỗi: {e}")
        time.sleep(0.1)
    print("Hoàn tất kịch bản giả lập!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["normal", "gradual_drift", "sudden_shift"], default="normal")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--evidently-url", default="http://localhost:8001")
    args = parser.parse_args()
    run(args.scenario, args.iterations, args.api_url, args.evidently_url)