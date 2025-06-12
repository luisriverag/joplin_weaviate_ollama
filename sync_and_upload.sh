#!/bin/bash

# Load environment variables
set -a
source .env
set +a

echo "🔄 Syncing Joplin notes + 📤 Uploading to Weaviate"
python3 joplin_sync.py --sync --upload --workers 8  --batch-size 500 --progress 