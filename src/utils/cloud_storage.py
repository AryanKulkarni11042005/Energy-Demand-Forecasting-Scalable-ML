"""S3-compatible Cloud Storage Client for Models and Monitoring Reports (Phase 11).

Compatible with:
  1. Cloudflare R2 (Recommended: 10 GB permanently free, $0 egress fees)
  2. AWS S3 (12-month free tier)
  3. Local filesystem fallback (when credentials are unset or running offline)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("eia.cloud_storage")


class CloudStorageClient:
    def __init__(
        self,
        endpoint_url: str | None = None,
        bucket_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region_name: str = "auto",
    ) -> None:
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL")
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", "energy-demand-models")
        self.access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "auto")
        self._client: Any = None

    @property
    def is_configured(self) -> bool:
        return bool(self.access_key_id and self.secret_access_key and self.bucket_name)

    def _get_boto_client(self) -> Any:
        if self._client is None:
            if not self.is_configured:
                raise ValueError("Cloud storage credentials not configured.")
            import boto3

            client_kwargs: dict[str, Any] = {
                "aws_access_key_id": self.access_key_id,
                "aws_secret_access_key": self.secret_access_key,
                "region_name": self.region_name,
            }
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url

            self._client = boto3.client("s3", **client_kwargs)
        return self._client

    def upload_file(self, local_path: Path, remote_key: str) -> bool:
        """Upload a local file to remote storage. Returns False if unconfigured or failed."""
        if not self.is_configured:
            logger.info("Cloud storage unconfigured; skipping remote upload of %s", local_path)
            return False
        try:
            client = self._get_boto_client()
            client.upload_file(str(local_path), self.bucket_name, remote_key)
            logger.info("Uploaded %s to s3://%s/%s", local_path, self.bucket_name, remote_key)
            return True
        except Exception as e:
            logger.warning("Failed to upload %s to s3: %s", local_path, e)
            return False

    def download_file(self, remote_key: str, local_path: Path) -> bool:
        """Download a remote key to local path. Returns False if unconfigured or missing."""
        if not self.is_configured:
            logger.info("Cloud storage unconfigured; skipping remote download of %s", remote_key)
            return False
        try:
            client = self._get_boto_client()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(self.bucket_name, remote_key, str(local_path))
            logger.info("Downloaded s3://%s/%s to %s", self.bucket_name, remote_key, local_path)
            return True
        except Exception as e:
            logger.warning("Failed to download s3://%s/%s: %s", self.bucket_name, remote_key, e)
            return False

    def sync_models_to_remote(self, models_dir: Path) -> int:
        """Upload all joblib models to cloud storage."""
        count = 0
        for p in models_dir.glob("*.joblib"):
            if self.upload_file(p, f"models/{p.name}"):
                count += 1
        return count

    def sync_models_from_remote(self, models_dir: Path) -> int:
        """Download all 3 tier models from cloud storage."""
        count = 0
        for tier in ["large", "medium", "small"]:
            filename = f"model_tier_{tier}.joblib"
            target = models_dir / filename
            if self.download_file(f"models/{filename}", target):
                count += 1
        return count
