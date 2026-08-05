import os
import io
import uuid
import logging
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

# Configure logging to print to the console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gdrive_sync")

# Load credentials from .env file
load_dotenv()

# Try importing backend components (for this project).
# If they fail to import, we fallback to a standalone mode so ANYONE can run it anywhere!
try:
    from backend.storage.postgres_store import PostgresStore
    from backend.api.main import _run_ingestion, _file_type, UPLOAD_DIR
    HAS_BACKEND = True
    logger.info("Project backend components imported successfully. Running in RAG Ingestion mode.")
except ImportError:
    HAS_BACKEND = False
    UPLOAD_DIR = "./gdrive_downloads"
    
    def _file_type(name: str) -> str:
        """Fallback helper to extract file extension if backend is missing."""
        return name.split(".")[-1].lower() if "." in name else "unknown"
        
    logger.info("Running in standalone utility mode (no local backend detected).")


def get_gdrive_service() -> build:
    """Acquires Google Drive API Service using Refresh Tokens."""
    client_id = os.getenv("GDRIVE_CLIENT_ID")
    client_secret = os.getenv("GDRIVE_CLIENT_SECRET")
    refresh_token = os.getenv("GDRIVE_REFRESH_TOKEN")

    if not all([client_id, client_secret, refresh_token]):
        raise ValueError(
            "Missing Google Drive Credentials (GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_REFRESH_TOKEN) in .env file."
        )

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/drive"]
    )

    # Refresh credentials if expired
    if not creds.valid:
        creds.refresh(Request())

    return build('drive', 'v3', credentials=creds)


def move_gdrive_file(service, file_id: str, current_parent_id: str, target_parent_id: str) -> None:
    """Moves a file on Google Drive by removing its current parent folder and adding target folder."""
    try:
        service.files().update(
            fileId=file_id,
            addParents=target_parent_id,
            removeParents=current_parent_id,
            fields='id, parents'
        ).execute()
        logger.info(f"Moved Drive file {file_id} to parent folder {target_parent_id}")
    except Exception as e:
        logger.error(f"Failed to move Drive file {file_id}: {e}")


def sync_and_ingest(custom_callback=None) -> dict:
    """
    Scans Google Drive watch folder, downloads files, processes/ingests them, and relocates files.
    
    Args:
        custom_callback: A function: `callback(local_file_path, filename) -> bool`
                         If provided, it overrides default ingestion logic.
    """
    watch_id = os.getenv("GDRIVE_WATCH_FOLDER_ID")
    processed_id = os.getenv("GDRIVE_PROCESSED_FOLDER_ID")
    failed_id = os.getenv("GDRIVE_FAILED_FOLDER_ID")

    if not watch_id:
        logger.error("GDRIVE_WATCH_FOLDER_ID is not configured in your .env file.")
        return {"error": "Missing watch folder ID"}

    try:
        service = get_gdrive_service()
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return {"error": f"Auth failed: {e}"}

    logger.info(f"Scanning watch folder (ID: {watch_id}) for files...")
    try:
        query = f"'{watch_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get("files", [])
    except Exception as e:
        logger.error(f"Failed to query files from watch folder: {e}")
        return {"error": f"Query failed: {e}"}

    if not files:
        logger.info("No files found in the watch folder.")
        return {"status": "success", "scanned": 0, "processed": []}

    ingested = []
    failed = []

    logger.info(f"Found {len(files)} files. Starting processing...")

    for file in files:
        file_id = file["id"]
        filename = file["name"]
        mime_type = file["mimeType"]

        # Skip folders and hidden system files
        if mime_type == "application/vnd.google-apps.folder" or filename.startswith("."):
            continue

        file_type = _file_type(filename)

        # Generate a unique document/file name
        document_id = str(uuid.uuid4())
        dest_path = os.path.join(UPLOAD_DIR, f"{document_id}_{filename}")
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        logger.info(f"Downloading {filename}...")
        try:
            # Download file contents
            request = service.files().get_media(fileId=file_id)
            with io.FileIO(dest_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            failed.append(filename)
            if failed_id:
                move_gdrive_file(service, file_id, watch_id, failed_id)
            continue

        # ----------------------------------------------------
        # Processing Logic (Dynamic based on running environment)
        # ----------------------------------------------------
        success = True

        if custom_callback:
            # Case 1: Custom callback was passed by an external developer
            try:
                success = custom_callback(dest_path, filename)
            except Exception as cb_err:
                logger.error(f"Custom callback failed for {filename}: {cb_err}")
                success = False
        
        elif HAS_BACKEND:
            # Case 2: Running inside your RAG Accelerator project
            logger.info(f"Registering {filename} in database...")
            pg = PostgresStore()
            try:
                pg.insert_document(document_id, filename, file_type, dest_path)
            except Exception as db_err:
                logger.error(f"DB insert failed for {filename}: {db_err}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                success = False
            finally:
                pg.close()

            if success:
                logger.info(f"Triggering RAG pipeline ingestion for {filename}...")
                try:
                    _run_ingestion(document_id, dest_path, file_type, filename)
                except Exception as ingest_err:
                    logger.error(f"Ingestion logic failed for {filename}: {ingest_err}")
                    # Let the status check determine if it succeeded
                
                # Check status
                status = "failed"
                pg = PostgresStore()
                try:
                    doc = pg.get_document(document_id)
                    if doc:
                        status = doc.get("status", "failed")
                except Exception as check_err:
                    logger.error(f"DB check failed: {check_err}")
                finally:
                    pg.close()

                success = (status == "ready")
        
        else:
            # Case 3: Running standalone on a clean machine (just downloads files)
            logger.info(f"Successfully saved {filename} locally to {dest_path}")
            success = True

        # ----------------------------------------------------
        # Post-Processing Folder Movement
        # ----------------------------------------------------
        if success:
            logger.info(f"Successfully processed {filename}.")
            ingested.append(filename)
            if processed_id:
                move_gdrive_file(service, file_id, watch_id, processed_id)
        else:
            logger.error(f"Failed to process {filename}.")
            failed.append(filename)
            if failed_id:
                move_gdrive_file(service, file_id, watch_id, failed_id)

    return {
        "status": "success",
        "scanned": len(files),
        "ingested": ingested,
        "failed": failed
    }


if __name__ == "__main__":
    logger.info("Executing sync loop...")
    results = sync_and_ingest()
    logger.info(f"Sync complete. Details: {results}")
