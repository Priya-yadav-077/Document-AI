#!/bin/bash
# Download Qasper dataset from GitHub

echo "Downloading Qasper dataset from GitHub..."

# Create data directory
mkdir -p data

cd data

# Download files from the original Qasper GitHub repo (raw content)
echo "Downloading training data..."
curl -L -o qasper-train-v0.3.json https://raw.githubusercontent.com/allenai/qasper/master/data/qasper-train-v0.3.json

echo "Downloading development data..."
curl -L -o qasper-dev-v0.3.json https://raw.githubusercontent.com/allenai/qasper/master/data/qasper-dev-v0.3.json

echo "Downloading test data..."
curl -L -o qasper-test-v0.3.json https://raw.githubusercontent.com/allenai/qasper/master/data/qasper-test-v0.3.json

cd ..

echo ""
echo "✅ Download complete!"
echo ""
echo "Verifying file sizes..."
ls -lh data/*.json
echo ""
echo "Files downloaded to:"
echo "  - data/qasper-train-v0.3.json"
echo "  - data/qasper-dev-v0.3.json"
echo "  - data/qasper-test-v0.3.json"
