# Standalone Google Drive Sync & Ingestion Script Guide

This guide explains how to use **`gdrive_sync.py`** to sync Google Drive files on-demand in any Python project. 

The script is **completely decoupled**—it operates in your project by default, but fallback logic allows anyone to copy this single file into *any* codebase and run it out-of-the-box.

---

## 1. File Location
The single-file script is located in your project root:
📄 **[gdrive_sync.py](file:///c:/Users/visha/OneDrive/Desktop/AI-Accelerator-vishal-new/gdrive_sync.py)**

---

## 2. Requirements & Setup

To use this script in any environment:

1. Install the required client libraries:
   ```bash
   pip install google-api-python-client google-auth python-dotenv
   ```
2. Set up the following environment variables in a `.env` file in the directory where you run the script:
   ```env
   # OAuth Credentials
   GDRIVE_CLIENT_ID=your_client_id
   GDRIVE_CLIENT_SECRET=your_client_secret
   GDRIVE_REFRESH_TOKEN=your_refresh_token
   
   # Folder IDs
   GDRIVE_WATCH_FOLDER_ID=your_watch_folder_id
   GDRIVE_PROCESSED_FOLDER_ID=your_processed_folder_id
   GDRIVE_FAILED_FOLDER_ID=your_failed_folder_id
   ```

---

## 3. How Anyone Can Run It

### Mode A: Standalone File Downloader (Out-of-the-box)
If this file is copied to a fresh directory with no local backend components, running:
```bash
python gdrive_sync.py
```
It will automatically run in **standalone utility mode**:
* It connects to your Google Drive folder.
* It downloads all new files to a local directory named `./gdrive_downloads`.
* Once downloaded successfully, it automatically relocates them to the `processed` folder on Google Drive.

### Mode B: Attaching Custom Ingestion Code (For other developers)
Other developers can easily import the script and supply their own custom file processing callback function. For example:

```python
from gdrive_sync import sync_and_ingest

def my_custom_processor(file_path: str, filename: str) -> bool:
    """
    Write your custom code here.
    Return True if the file was indexed successfully.
    Return False if it failed (it will be moved to Google Drive failed folder).
    """
    print(f"Indexing {filename} in my custom vector database...")
    try:
        # Write your database insert or parser code here
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

# Trigger the sync and pass your custom callback
report = sync_and_ingest(custom_callback=my_custom_processor)
print("Sync Report:", report)
```

### Mode C: RAG Ingestion Mode (For this project)
When run in your current workspace root:
```bash
python gdrive_sync.py
```
It automatically detects the backend models. It downloads files to `uploads/`, writes metadata records to PostgreSQL, runs docling-based parsing, chunks the text, embeds it with Nomic/Huggingface, and indexes it into Qdrant. It then moves the files on Drive depending on the status of the pipeline!
