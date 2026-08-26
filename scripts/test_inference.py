#!/usr/bin/env python3
"""
Test Inference Script for all images in data/test/

This script sends each test image to the model server, collects predictions,
and compares them with expected labels from metadata.csv.
"""

import json
import sys
import os
from pathlib import Path

import requests
import pandas as pd

# Configuration
MODEL_SERVER_URL = "http://localhost:5001/invocations"
TEST_DATA_DIR = Path("./data/test")
LOG_FILE = Path("./logs/inference_test_results.json")
SUMMARY_FILE = Path("./logs/inference_test_summary.txt")

# Class mapping (from config)
CLASS_NAMES = {0: "clean", 1: "defective"}


def check_server_health():
    """Check if the model server is running and healthy."""
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            return True
        else:
            print(f"Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error checking server health: {e}")
        return False


def load_metadata():
    """Load metadata.csv and return a dictionary mapping filename to expected class."""
    metadata_path = TEST_DATA_DIR / "metadata.csv"
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found at {metadata_path}")
        sys.exit(1)
    
    df = pd.read_csv(metadata_path)
    metadata = {}
    for _, row in df.iterrows():
        metadata[row['filename']] = {
            'class': int(row['class']),
            'class_name': row['class_name'],
            'impurity_type': row['impurity_type'],
        }
    return metadata


def send_inference_request(image_path):
    """Send an image as bytes to the model server and return the response."""
    # Read image as bytes
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None
    
    # MLflow model server expects: {"instances": [image_bytes]}
    # Using base64 encoding for JSON serialization
    import base64
    image_bytes_b64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {"instances": [image_bytes_b64]}
    
    try:
        response = requests.post(
            MODEL_SERVER_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending inference request for {image_path}: {e}")
        return None


def parse_prediction(response, expected_class):
    """
    Parse the model server response and extract the predicted class.
    
    The SimpleLoggingWrapper returns:
    - {"predictions": [0, 1, ...], "confidences": [...], "raw_outputs": [...], ...}
    
    MLflow pyfunc model server wraps the response in an outer dict, so we may receive:
    - {"predictions": {"predictions": [...], "confidences": [...], ...}}
    
    Parameters
    ----------
    response : dict
        The JSON response from the model server.
    expected_class : int
        The expected class for this sample (currently unused but kept for compatibility).
        
    Returns
    -------
    tuple
        (predicted_class, predicted_class_name, confidence) or (None, None, None) on error.
    """
    if response is None:
        return None, None, None
    
    try:
        # Check if MLflow has wrapped the response
        predictions = response.get("predictions", [])
        if isinstance(predictions, dict):
            # MLflow wrapped the response, unwrap it
            predictions = predictions.get("predictions", [])
            confidences = response.get("predictions", {}).get("confidences", [])
            raw_outputs = response.get("predictions", {}).get("raw_outputs", [])
            timestamp = response.get("predictions", {}).get("timestamp", "")
            num_samples = response.get("predictions", {}).get("num_samples", 0)
        else:
            # Direct response from model
            confidences = response.get("confidences", [])
            raw_outputs = response.get("raw_outputs", [])
            timestamp = response.get("timestamp", "")
            num_samples = response.get("num_samples", 0)
        
        if not predictions:
            print(f"  WARNING: Empty predictions in response: {response}")
            return None, None, None
        
        # Get the first prediction (for single image inference)
        if isinstance(predictions, list) and len(predictions) > 0:
            predicted_class = predictions[0]
            
            # Get confidence from confidences if available, otherwise from raw_outputs
            if confidences and len(confidences) > 0:
                confidence = confidences[0]
            elif raw_outputs and len(raw_outputs) > 0:
                confidence = raw_outputs[0]
            else:
                confidence = 1.0 if predicted_class is not None else None
            
            if predicted_class is not None:
                return predicted_class, CLASS_NAMES.get(predicted_class, "unknown"), confidence
        
        print(f"  WARNING: Unable to parse predictions: {predictions}")
        return None, None, None
        
    except Exception as e:
        print(f"  ERROR: Failed to parse response: {e}")
        print(f"  Response: {response}")
        return None, None, None


def test_all_images():
    """Test inference on all images in the test directory."""
    # Check server health
    print("Checking model server health...")
    if not check_server_health():
        print("ERROR: Model server is not running at http://localhost:5001")
        print("Start it first with: ./scripts/start_model_server.sh")
        sys.exit(1)
    
    print("Model server is running. Loading metadata...\n")
    
    # Load metadata
    metadata = load_metadata()
    
    # Find all test images
    test_images = list(TEST_DATA_DIR.glob("*.jpg"))
    test_images.sort()
    
    if not test_images:
        print(f"ERROR: No test images found in {TEST_DATA_DIR}")
        sys.exit(1)
    
    print(f"Found {len(test_images)} test images\n")
    
    # Initialize results
    results = []
    correct = 0
    incorrect = 0
    
    # Create logs directory
    Path("./logs").mkdir(parents=True, exist_ok=True)
    
    # Test each image
    print("=" * 70)
    print("Testing Inference")
    print("=" * 70)
    
    for image_path in test_images:
        filename = image_path.name
        
        # Get expected class from metadata
        expected_info = metadata.get(filename)
        if expected_info is None:
            print(f"WARNING: No metadata found for {filename}, skipping...")
            continue
        
        expected_class = expected_info['class']
        expected_class_name = expected_info['class_name']
        impurity_type = expected_info['impurity_type']
        
        print(f"\nTesting: {filename}")
        print(f"  Expected: class={expected_class} ({expected_class_name}), impurity={impurity_type}")
        
        # Send inference request
        response = send_inference_request(image_path)
        
        if response is None:
            print(f"  Status: FAILED")
            results.append({
                'filename': filename,
                'expected_class': expected_class,
                'expected_class_name': expected_class_name,
                'impurity_type': impurity_type,
                'predicted_class': None,
                'predicted_class_name': None,
                'confidence': None,
                'correct': False,
                'error': 'Request failed'
            })
            continue
        
        # Parse prediction
        predicted_class, predicted_class_name, confidence = parse_prediction(response, expected_class)
        
        if predicted_class is None:
            print(f"  Status: FAILED (parse error)")
            print(f"  Response: {response}")
            results.append({
                'filename': filename,
                'expected_class': expected_class,
                'expected_class_name': expected_class_name,
                'impurity_type': impurity_type,
                'predicted_class': None,
                'predicted_class_name': None,
                'confidence': None,
                'correct': False,
                'error': 'Parse failed',
                'raw_response': response
            })
            continue
        
        # Determine if correct
        is_correct = (predicted_class == expected_class)
        if is_correct:
            correct += 1
            status = "CORRECT"
        else:
            incorrect += 1
            status = "INCORRECT"
        
        print(f"  Predicted: class={predicted_class} ({predicted_class_name}), confidence={confidence:.4f}")
        print(f"  Status: {status}")
        
        results.append({
            'filename': filename,
            'expected_class': expected_class,
            'expected_class_name': expected_class_name,
            'impurity_type': impurity_type,
            'predicted_class': predicted_class,
            'predicted_class_name': predicted_class_name,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # Calculate accuracy
    total = correct + incorrect
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Total images tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Save results to JSON
    with open(LOG_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary to text file
    with open(SUMMARY_FILE, 'w') as f:
        f.write("Inference Test Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model Server: {MODEL_SERVER_URL}\n")
        f.write(f"Test Data: {TEST_DATA_DIR}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Total: {total}\n")
        f.write(f"  Correct: {correct}\n")
        f.write(f"  Incorrect: {incorrect}\n")
        f.write(f"  Accuracy: {accuracy:.2f}%\n")
        f.write(f"\nDetailed results saved to: {LOG_FILE}\n")
    
    print(f"\nDetailed results saved to: {LOG_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    
    return accuracy


if __name__ == "__main__":
    accuracy = test_all_images()
    sys.exit(0 if accuracy >= 0 else 1)
