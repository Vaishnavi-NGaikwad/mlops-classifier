#!/bin/bash
# Smoke Test Script - M4 Post-Deploy Verification

BASE_URL="http://localhost:5000"
PASS=0
FAIL=0

echo "============================================"
echo "  Smoke Tests - Pet Classifier Service"
echo "============================================"

# Wait for service to be ready - retry until health returns 200
echo "Waiting for service to be ready..."
MAX_WAIT=300  # max 5 minutes
INTERVAL=10
ELAPSED=0

until [ "$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")" -eq 200 ]; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Service did not become ready after ${MAX_WAIT}s. Exiting."
        exit 1
    fi
    echo "Service not ready yet (${ELAPSED}s elapsed)... retrying in ${INTERVAL}s"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "Service is ready after ${ELAPSED}s!"

# ─────────────────────────────────────────────
# TEST 1: Health Check
# ─────────────────────────────────────────────
echo ""
echo "[TEST 1] Health Check Endpoint..."

HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")

if [ "$HEALTH_RESPONSE" -eq 200 ]; then
    echo "PASS: Health check returned 200 OK"
    PASS=$((PASS + 1))
else
    echo "FAIL: Health check returned $HEALTH_RESPONSE (expected 200)"
    FAIL=$((FAIL + 1))
fi

# ─────────────────────────────────────────────
# TEST 2: Health Response Body
# ─────────────────────────────────────────────
echo ""
echo "[TEST 2] Health Response contains 'healthy' status..."

HEALTH_BODY=$(curl -s "$BASE_URL/health")

if echo "$HEALTH_BODY" | grep -q "healthy"; then
    echo "PASS: Health response contains 'healthy'"
    PASS=$((PASS + 1))
else
    echo "FAIL: Health response missing 'healthy': $HEALTH_BODY"
    FAIL=$((FAIL + 1))
fi

# ─────────────────────────────────────────────
# TEST 3: Prediction Endpoint
# ─────────────────────────────────────────────
echo ""
echo "[TEST 3] Prediction Endpoint..."

python3 -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
img.save('/tmp/test_image.jpg')
print('Test image created')
"

PREDICT_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST \
    -F "file=@/tmp/test_image.jpg" \
    "$BASE_URL/predict")

if [ "$PREDICT_RESPONSE" -eq 200 ]; then
    echo "PASS: Prediction endpoint returned 200 OK"
    PASS=$((PASS + 1))
else
    echo "FAIL: Prediction endpoint returned $PREDICT_RESPONSE (expected 200)"
    FAIL=$((FAIL + 1))
fi

# ─────────────────────────────────────────────
# TEST 4: Prediction Response Fields
# ─────────────────────────────────────────────
echo ""
echo "[TEST 4] Prediction response contains required fields..."

PREDICT_BODY=$(curl -s \
    -X POST \
    -F "file=@/tmp/test_image.jpg" \
    "$BASE_URL/predict")

if echo "$PREDICT_BODY" | grep -q "prediction" && echo "$PREDICT_BODY" | grep -q "confidence"; then
    echo "PASS: Prediction response contains 'prediction' and 'confidence'"
    PASS=$((PASS + 1))
else
    echo "FAIL: Prediction response missing fields: $PREDICT_BODY"
    FAIL=$((FAIL + 1))
fi

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Smoke Test Results"
echo "============================================"
echo "  PASSED: $PASS"
echo "  FAILED: $FAIL"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    echo "SMOKE TESTS FAILED - Deployment verification unsuccessful"
    exit 1
else
    echo "ALL SMOKE TESTS PASSED - Deployment successful!"
    exit 0
fi