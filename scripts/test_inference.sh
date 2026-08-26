#!/bin/bash

# Test Inference Script for all images in data/test/
# This script sends each test image to the model server and compares predictions with expected labels

MODEL_SERVER_URL="http://localhost:5001/invoke"
TEST_DATA_DIR="./data/test"
LOG_FILE="./logs/inference_test_results.log"

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Counters for summary
TOTAL=0
CORRECT=0
INCORRECT=0

# Print header
echo "========================================"
echo "Testing Inference on data/test/ images"
echo "========================================"
echo ""
echo "Model Server: ${MODEL_SERVER_URL}"
echo "Test Data Directory: ${TEST_DATA_DIR}"
echo ""

# Write log header
echo "Inference Test Results - $(date)" > "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Check if model server is running
if ! curl -s "http://localhost:5001/health" > /dev/null 2>&1; then
    echo "ERROR: Model server is not running at http://localhost:5001"
    echo "Start it first with: ./scripts/start_model_server.sh"
    exit 1
fi

echo "Model server is running. Starting inference tests..."
echo "" >> "$LOG_FILE"

# Process each image file in the test directory
for IMAGE_FILE in "${TEST_DATA_DIR}"/*.jpg; do
    # Skip if no jpg files found
    [ -e "$IMAGE_FILE" ] || continue
    
    TOTAL=$((TOTAL + 1))
    FILENAME=$(basename "$IMAGE_FILE")
    
    echo "Processing: ${FILENAME}"
    
    # Extract expected label from metadata.csv
    # Format: filename,class,class_name,...
    EXPECTED_LABEL=$(grep "^${FILENAME}," "${TEST_DATA_DIR}/metadata.csv" | cut -d',' -f2)
    EXPECTED_CLASS=$(grep "^${FILENAME}," "${TEST_DATA_DIR}/metadata.csv" | cut -d',' -f3)
    
    if [ -z "$EXPECTED_LABEL" ]; then
        echo "  WARNING: No metadata found for ${FILENAME}, skipping..."
        continue
    fi
    
    echo "  Expected: class=${EXPECTED_LABEL} (${EXPECTED_CLASS})"
    
    # Send image path to model server
    # Using the MLflow model server format: {"data": ["path/to/image.jpg"]}
    RESPONSE=$(curl -s -X POST "${MODEL_SERVER_URL}" \
        -H "Content-Type: application/json" \
        -d "{\"data\": [\"${IMAGE_FILE}\"]}" 2>&1)
    
    # Check if curl succeeded
    if [ $? -ne 0 ]; then
        echo "  ERROR: Failed to send request for ${FILENAME}"
        echo "  Response: ${RESPONSE}"
        echo "" >> "$LOG_FILE"
        continue
    fi
    
    # Parse prediction from response
    # MLflow returns: {"predictions": [predictions...]}
    # For image classification, predictions should be a 2D array
    PREDICTION=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(json.dumps(data))")
    
    echo "  Response: ${RESPONSE}"
    
    # Log the result
    echo "File: ${FILENAME}" >> "$LOG_FILE"
    echo "  Expected: ${EXPECTED_LABEL} (${EXPECTED_CLASS})" >> "$LOG_FILE"
    echo "  Response: ${RESPONSE}" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # For now, count as correct if we got a response (detailed comparison needs Python)
    echo "  Status: RECEIVED"
    echo ""
done

# Print summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total images tested: ${TOTAL}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Note: For detailed accuracy metrics, use the Python version of this script."
