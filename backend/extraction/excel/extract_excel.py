import os
import uuid
import json
import pandas as pd
from typing import List, Dict, Any, Optional

def extract_excel(file_path: str, document_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Reads an Excel file, cleans empty space, and returns data
    strictly matching the v8.3 NormalizedBlock contract.
    Utilizes a config dictionary to avoid hardcoded metadata.
    """
    blocks: List[Dict[str, Any]] = []
    cfg = config or {}
    
    # Standardize IDs and filename
    doc_id = str(document_id) if document_id else str(uuid.uuid4())
    filename = os.path.basename(file_path)
    
    try:
        excel_data = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    except Exception as e:
        print(f"Failed to read file {file_path}: {e}")
        return blocks
    
    for sheet_name, df in excel_data.items():
        try:
            # Cleans all empty rows and columns
            df_cleaned = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df_cleaned.empty:
                continue
                
            df_cleaned = df_cleaned.fillna("")
            
            headers = [str(col) for col in df_cleaned.columns.tolist()]
            rows = df_cleaned.values.tolist()
            
            # 8.3 SCHEMA
            block = {
                "block_id": str(uuid.uuid4()),
                "document_id": doc_id,
                "type": "table",
                "text": None,
                "table_data": {
                    "headers": headers,
                    "rows": rows
                },
                "source_ref": {
                    "filename": filename,
                    "page": None,
                    "sheet": str(sheet_name),
                    "slide": None,
                    "bbox": None
                },
                "confidence": cfg.get("extraction_confidence", 1.0),
                "language": cfg.get("default_language", "en"),
                "metadata": {
                    "enrichment_failed": cfg.get("enrichment_failed_flag", False)
                }
            }
            
            blocks.append(block)
            
        except Exception as e:
            # Fail gracefully
            print(f"Skipping sheet '{sheet_name}' due to error: {e}")
            continue
            
    return blocks

# ---------------------------------------------------------
# SANDBOX TEST BLOCK 
# ---------------------------------------------------------
if __name__ == "__main__":
    test_file_path = "test-data/test.xlsx"  
    
    # Simulation of a pipeline config being passed in
    mock_pipeline_config = {
        "extraction_confidence": 0.95,
        "default_language": "en"
    }
    
    print(f"Testing STRICT 8.3 extraction on {test_file_path}...\n")
    if os.path.exists(test_file_path):
        results = extract_excel(test_file_path, config=mock_pipeline_config)
        
        if results:
            print(f"✅ Success! Extracted {len(results)} table(s).\n")
            print("--- Output Preview ---")
            print(json.dumps(results[0], indent=2))
        else:
            print("❌ No tables extracted.")
    else:
        print(f"⚠️ Dummy file '{test_file_path}' not found. Check test-data folder!")