#!/bin/bash

# Load environment variables
set -a
source .env
set +a

echo "🔄 Syncing Joplin notes..."
python3 joplin_sync.py --sync

echo "📤 Uploading to Weaviate..."
python3 joplin_sync.py --upload
