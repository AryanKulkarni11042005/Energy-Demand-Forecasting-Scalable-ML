"""Unit tests for CloudStorageClient (Phase 11).

Runs 100% offline using mocks and local temporary paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.cloud_storage import CloudStorageClient


class TestCloudStorageClientOffline:
    def test_unconfigured_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("S3_BUCKET_NAME", raising=False)

        client = CloudStorageClient()
        assert not client.is_configured
        assert client.upload_file(Path("dummy.txt"), "dummy.txt") is False
        assert client.download_file("dummy.txt", Path("dummy.txt")) is False

    def test_configured_when_env_present(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "mock_key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "mock_secret")
        monkeypatch.setenv("S3_BUCKET_NAME", "mock_bucket")

        client = CloudStorageClient()
        assert client.is_configured

    @patch("boto3.client")
    def test_upload_success(self, mock_boto: MagicMock, tmp_path: Path):
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        dummy_file = tmp_path / "model.joblib"
        dummy_file.write_bytes(b"dummy model data")

        client = CloudStorageClient(
            access_key_id="test_key",
            secret_access_key="test_secret",
            bucket_name="test_bucket",
        )

        assert client.upload_file(dummy_file, "models/model.joblib") is True
        mock_s3.upload_file.assert_called_once_with(
            str(dummy_file), "test_bucket", "models/model.joblib"
        )

    @patch("boto3.client")
    def test_download_success(self, mock_boto: MagicMock, tmp_path: Path):
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        target_file = tmp_path / "downloaded" / "model.joblib"

        client = CloudStorageClient(
            access_key_id="test_key",
            secret_access_key="test_secret",
            bucket_name="test_bucket",
        )

        assert client.download_file("models/model.joblib", target_file) is True
        mock_s3.download_file.assert_called_once_with(
            "test_bucket", "models/model.joblib", str(target_file)
        )

    @patch("boto3.client")
    def test_sync_models_to_remote(self, mock_boto: MagicMock, tmp_path: Path):
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        (tmp_path / "model_tier_large.joblib").write_bytes(b"large")
        (tmp_path / "model_tier_medium.joblib").write_bytes(b"medium")
        (tmp_path / "model_tier_small.joblib").write_bytes(b"small")

        client = CloudStorageClient(
            access_key_id="test_key",
            secret_access_key="test_secret",
            bucket_name="test_bucket",
        )

        synced = client.sync_models_to_remote(tmp_path)
        assert synced == 3
        assert mock_s3.upload_file.call_count == 3
