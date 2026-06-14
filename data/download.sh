#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

wget -O wiki_v2.entities.jsonl.zst https://s3.eu-central-2.wasabisys.com/ailab-public/models/lvnel/wiki_v2.entities.jsonl.zst
zstd -d -f wiki_v2.entities.jsonl.zst -o wiki_v2.entities.jsonl
