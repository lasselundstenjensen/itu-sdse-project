from src.deployment.drift_detection import load_inference_logs, load_baseline_stats, detect_drift
import requests
artifact_dir = "artifacts"
inference_log_path = artifact_dir + "/inference_log.json"

def test_drift(drift_default_days,):
    logs = load_inference_logs(inference_log_path)
    print(f"Logged requests: {len(logs)}")
    print(logs[-3:])

    baseline = load_baseline_stats()
    drift_results = detect_drift(logs, days=drift_default_days, baseline_stats=baseline)
    print("\nDrift check results:")
    print(drift_results)

    if drift_results["overall_drift_detected"]:
        print("\n!! Drift detected - time to retrain and redeploy")
    else:
        print("\nNo drift detected")

    # The endpoint does the same check over HTTP
    print("\nGET /drift ->")
    print(requests.get(f"http://localhost:{port}/drift").json())

if __name__ == "__main__":
    test_drift()