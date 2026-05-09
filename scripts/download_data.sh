#!/bin/bash
# Download pre-built index from GitHub Releases.
# Usage: bash scripts/download_data.sh

set -e
cd "$(dirname "$0")/.."

REPO="ton-user/rag-wikipedia"   # ← à remplacer par ton vrai repo
TAG="v1.0"
FILE="data-prebuilt.tar.gz"
URL="https://github.com/${REPO}/releases/download/${TAG}/${FILE}"

echo "=== Downloading pre-built index ==="
echo "From: $URL"

if command -v wget &>/dev/null; then
  wget -q --show-progress -O "$FILE" "$URL"
elif command -v curl &>/dev/null; then
  curl -L --progress-bar -o "$FILE" "$URL"
else
  echo "Error: wget or curl required"
  exit 1
fi

echo "Extracting…"
tar -xzf "$FILE"
rm "$FILE"

echo ""
echo "✓ Done! Index ready."
echo "  Run: streamlit run app/main.py"
