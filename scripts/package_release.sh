#!/bin/bash
# Run this ONCE on your machine to create the release archive.
# Usage: bash scripts/package_release.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Packaging data/ for GitHub Release ==="

# Exclude cache, logs, feedback (personal/ephemeral data)
tar -czf data-prebuilt.tar.gz \
  data/raw/articles.json \
  data/raw/arxiv_papers.json.gz \
  data/indexes/faiss.index \
  data/indexes/faiss_map.pkl \
  data/indexes/bm25.pkl \
  data/indexes/chunks.pkl

SIZE=$(du -sh data-prebuilt.tar.gz | cut -f1)
echo "✓ Created data-prebuilt.tar.gz ($SIZE)"
echo ""
echo "Next steps:"
echo "  1. Go to GitHub → your repo → Releases → Draft a new release"
echo "  2. Tag: v1.0"
echo "  3. Upload: data-prebuilt.tar.gz"
echo "  4. Publish"
