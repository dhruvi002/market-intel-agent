#!/usr/bin/env python3
"""Bootstrap MinIO: create the sec-filings bucket if it does not already exist.

Usage (from project root):
    uv run python scripts/init_minio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path for uv workspace imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mia_ingestion.storage.minio_client import MinIOClient


def main() -> None:
    client = MinIOClient()
    client.ensure_bucket()
    settings_info = client._bucket
    print(f"MinIO bootstrap complete — bucket: {settings_info!r}")


if __name__ == "__main__":
    main()
