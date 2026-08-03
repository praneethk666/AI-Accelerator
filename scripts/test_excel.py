"""Test script to test Excel extraction and query tools.

This script runs:
1. ExcelExtractorTool (from backend/extraction/excel/tool.py)
   Extracts NormalizedBlocks containing the text and raw table chunks,
   saving them to output-2/extractor_text.txt and output-2/extractor_blocks.json.

2. ExcelTool (from backend/extraction/excel/excel_tool.py)
   Runs code against the Excel data sheets,
   saving the query execution output to output-2/interpreter_result.txt.
"""

import os
import sys
import json

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Module-level imports are kept minimal so Windows multiprocessing doesn't import them in child processes.


def main():
    # Target file
    excel_file = "Fiber BOQ.xlsx"
    if not os.path.exists(excel_file):
        print(f"Error: {excel_file} not found in the root directory.")
        sys.exit(1)

    print(f"Target Excel file found: {os.path.abspath(excel_file)}")

    # Ensure output-2 directory exists
    output_dir = "output-2"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory initialized: {os.path.abspath(output_dir)}\n")

    # =========================================================================
    # PART 1: Run ExcelExtractorTool (tool.py)
    # =========================================================================
    print("--- 1. Running ExcelExtractorTool (tool.py) ---")
    from backend.extraction.excel.tool import ExcelExtractorTool
    extractor = ExcelExtractorTool()
    
    mock_state = {
        "file_path": excel_file,
        "document_id": "test-doc-excel-extractor"
    }
    mock_config = {
        "extraction_confidence": 1.0,
        "default_language": "en"
    }

    try:
        extracted_state = extractor.run(mock_state, mock_config)
        blocks = extracted_state.get("blocks", [])
        
        print(f"Successfully extracted {len(blocks)} blocks/chunks.")

        # Save blocks JSON
        blocks_path = os.path.join(output_dir, "extractor_blocks.json")
        with open(blocks_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2)
        print(f"Saved chunks metadata to: {blocks_path}")

        # Save normal text (joined markdown text of all blocks)
        normal_text = "\n\n".join([f"=== Block: {b.get('type')} ===\n{b.get('text', '')}" for b in blocks])
        text_path = os.path.join(output_dir, "extractor_text.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(normal_text)
        print(f"Saved normal text to: {text_path}")

    except Exception as e:
        print(f"Failed running ExcelExtractorTool: {e}")

    print()

    # =========================================================================
    # PART 2: Run ExcelTool (excel_tool.py)
    # =========================================================================
    print("--- 2. Running ExcelTool (excel_tool.py) ---")
    from backend.extraction.excel.excel_tool import ExcelTool
    interpreter = ExcelTool()

    # Python code to count rows, look at headers, and compute sheet statistics
    code_query = """
# Let's inspect the active dataframe (df) and all available sheets (dfs)
sheet_names = list(dfs.keys())
print(f"Found sheets: {sheet_names}")
summary = {}

for name, sheet_df in dfs.items():
    print(f"Processing sheet: '{name}' with {len(sheet_df)} rows and {len(sheet_df.columns)} columns.")
    summary[name] = {
        "rows": len(sheet_df),
        "columns": list(sheet_df.columns),
        "head_preview": sheet_df.head(3).to_dict(orient="records")
    }

# Assign to final result variable
result = {
    "available_sheets": sheet_names,
    "sheet_summaries": summary
}
"""

    try:
        res = interpreter.run(
            filename_or_id=excel_file,
            code=code_query,
            sheet_name="all"
        )
        
        if res.get("success"):
            print("Successfully executed interpreter code.")
            
            # Save query result
            result_path = os.path.join(output_dir, "interpreter_result.txt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(str(res.get("result", "")))
            print(f"Saved interpreter result to: {result_path}")

            # Save stdout if any
            stdout_path = os.path.join(output_dir, "interpreter_stdout.txt")
            with open(stdout_path, "w", encoding="utf-8") as f:
                f.write(res.get("stdout", ""))
            print(f"Saved interpreter stdout to: {stdout_path}")
        else:
            print(f"Interpreter returned failure: {res.get('error')}")

    except Exception as e:
        print(f"Failed running ExcelTool: {e}")

    print("\nAll done! Check the output-2 folder for results.")

if __name__ == "__main__":
    main()
