from __future__ import annotations

import logging
import mimetypes
import os
import requests

logger = logging.getLogger(__name__)


def resolve_env_ref(val: str | None) -> str | None:
    """Resolve value from environment if it is formatted as ${VAR}."""
    if not val:
        return None
    val_str = str(val).strip()
    if val_str.startswith("${") and val_str.endswith("}"):
        var_name = val_str[2:-1]
        return os.getenv(var_name)
    return val_str


def get_supabase_config(config: dict | None = None) -> tuple[str, str, str]:
    """Retrieve Supabase URL, Key, and Bucket from config or env."""
    from dotenv import load_dotenv
    load_dotenv()

    storage_cfg = {}
    if config:
        storage_cfg = config.get("storage", {})

    url = resolve_env_ref(storage_cfg.get("supabase_url")) or os.getenv("SUPABASE_URL")
    key = resolve_env_ref(storage_cfg.get("supabase_key")) or os.getenv("SUPABASE_KEY")
    bucket = resolve_env_ref(storage_cfg.get("supabase_bucket")) or os.getenv("SUPABASE_BUCKET", "documents")

    if url:
        url = url.strip().rstrip("/")
        if "/rest/v1" in url:
            url = url.split("/rest/v1")[0].rstrip("/")

    return url or "", key or "", bucket or "documents"


def upload_to_supabase(
    bucket: str, key: str, file_path_or_bytes: str | bytes, config: dict | None = None
) -> str:
    """Upload a file or raw bytes to a Supabase storage bucket.

    Returns:
        The supabase URI (e.g. supabase://bucket/key)
    """
    url, api_key, resolved_bucket = get_supabase_config(config)
    bucket = resolve_env_ref(bucket) or resolved_bucket

    if not url or url.startswith("${"):
        raise ValueError("Supabase Project URL is not configured. Please set SUPABASE_URL in your .env file.")
    if not api_key or api_key.startswith("${"):
        raise ValueError("Supabase API/Service Key is not configured. Please set SUPABASE_KEY in your .env file.")
    if not bucket or bucket.startswith("${"):
        raise ValueError("Supabase Storage Bucket is not configured. Please set SUPABASE_BUCKET in your .env file.")

    url = url.rstrip("/")
    # Clean double slashes in object path but preserve bucket/key structure
    clean_key = "/".join(part for part in key.split("/") if part)
    upload_url = f"{url}/storage/v1/object/{bucket}/{clean_key}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "apikey": api_key,
    }

    mime_type, _ = mimetypes.guess_type(clean_key)
    if mime_type:
        headers["Content-Type"] = mime_type

    if isinstance(file_path_or_bytes, str):
        with open(file_path_or_bytes, "rb") as f:
            data = f.read()
    else:
        data = file_path_or_bytes

    logger.info("Uploading file to Supabase Storage: %s/%s", bucket, clean_key)
    response = requests.post(upload_url, headers=headers, data=data)

    if response.status_code == 409:
        logger.info("File %s already exists. Overwriting via PUT.", clean_key)
        response = requests.put(upload_url, headers=headers, data=data)

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Failed to upload file to Supabase: {response.text} (Status: {response.status_code})"
        )

    return f"supabase://{bucket}/{clean_key}"


def download_from_supabase(
    bucket: str, key: str, local_dest: str, config: dict | None = None
) -> None:
    """Download a file from Supabase storage bucket to a local path."""
    url, api_key, resolved_bucket = get_supabase_config(config)
    bucket = resolve_env_ref(bucket) or resolved_bucket

    if not url or url.startswith("${"):
        raise ValueError("Supabase Project URL is not configured. Please set SUPABASE_URL in your .env file.")
    if not api_key or api_key.startswith("${"):
        raise ValueError("Supabase API/Service Key is not configured. Please set SUPABASE_KEY in your .env file.")
    if not bucket or bucket.startswith("${"):
        raise ValueError("Supabase Storage Bucket is not configured. Please set SUPABASE_BUCKET in your .env file.")

    url = url.rstrip("/")
    clean_key = "/".join(part for part in key.split("/") if part)
    download_url = f"{url}/storage/v1/object/authenticated/{bucket}/{clean_key}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "apikey": api_key,
    }

    logger.info("Downloading file from Supabase Storage: %s/%s -> %s", bucket, clean_key, local_dest)
    response = requests.get(download_url, headers=headers, stream=True)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download file from Supabase: {response.text} (Status: {response.status_code})"
        )

    # Ensure parent directories exist
    os.makedirs(os.path.dirname(local_dest), exist_ok=True)
    with open(local_dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def delete_from_supabase(bucket: str, key: str, config: dict | None = None) -> None:
    """Delete a file from Supabase storage bucket."""
    url, api_key, resolved_bucket = get_supabase_config(config)
    bucket = resolve_env_ref(bucket) or resolved_bucket

    if not url or url.startswith("${") or not api_key or api_key.startswith("${"):
        logger.warning("Supabase credentials missing or unresolved; skipping file deletion from storage.")
        return

    url = url.rstrip("/")
    clean_key = "/".join(part for part in key.split("/") if part)
    delete_url = f"{url}/storage/v1/object/{bucket}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "apikey": api_key,
        "Content-Type": "application/json",
    }

    logger.info("Deleting file from Supabase Storage: %s/%s", bucket, clean_key)
    response = requests.delete(delete_url, headers=headers, json={"prefixes": [clean_key]})

    if response.status_code not in (200, 204):
        logger.error(
            "Failed to delete file from Supabase: %s (Status: %s)",
            response.text,
            response.status_code,
        )
