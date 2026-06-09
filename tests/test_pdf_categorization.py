#!/usr/bin/env python
"""Quick test script to verify PDF categorization works correctly."""

import sys
import os
import json
from pathlib import Path

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.categorize.categorize_tool import CategorizeTool


def get_test_config() -> dict:
    """Return a minimal test configuration."""
    return {
        "categorization": {
            "type_to_route": {
                "cad_drawing": "cad_route",
                "circuit_diagram": "circuit_route",
                "datasheet": "diagram_heavy",
                "report": "text_default",
                "invoice": "table_heavy",
                "presentation": "text_default",
                "spreadsheet": "table_heavy",
                "image": "image_route",
                "unknown": "text_default",
            },
            "confidence_thresholds": {
                "categorization_low_confidence": 0.5
            },
            "industry_keywords": {
                "automotive": ["toyota", "ford", "bmw", "vehicle", "engine", "torque", "transmission"],
                "electronics": ["circuit", "pcb", "schematic", "voltage", "resistor", "capacitor"],
                "manufacturing": ["assembly", "drawing", "tolerance", "weld", "machining"],
                "finance": ["invoice", "balance sheet", "revenue", "profit"],
                "legal": ["contract", "agreement", "clause", "liability"],
                "healthcare": ["patient", "diagnosis", "clinical", "pharma"],
                "general": []
            }
        },
        "deployment": {
            "default_industry": "electronics"
        },
        "vision": {
            "provider": "google",
            "model": "gemma-3-27b-it"
        }
    }


def test_pdf():
    """Test categorization on the electric circuit diagram PDF."""
    
    # Get config
    config = get_test_config()
    
    # Get the repo root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # PDF file path
    pdf_path = os.path.join(repo_root, "test-data", "Electric circuit diagram(MAIN).pdf")
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}")
        print(f"Absolute path: {os.path.abspath(pdf_path)}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in test-data:")
        test_data_dir = os.path.join(repo_root, "test-data")
        if os.path.exists(test_data_dir):
            for f in os.listdir(test_data_dir):
                print(f"  - {f}")
        return
    
    # Create state
    state = {
        "file_path": pdf_path
    }
    
    print(f"Testing PDF: {pdf_path}")
    print("=" * 80)
    
    # Run categorization
    tool = CategorizeTool()
    result = tool.run(state, config)
    
    # Pretty print results
    print("CATEGORIZATION RESULT:")
    print("=" * 80)
    print(json.dumps(state, indent=2))
    print("=" * 80)
    
    # Verify expected fields
    expected_fields = ["route", "document_type", "industry", "confidence", "reasoning", "errors"]
    print("\nVERIFICATION:")
    print("-" * 80)
    
    for field in expected_fields:
        if field in state:
            value = state[field]
            if isinstance(value, str) and len(value) > 100:
                print(f"✓ {field}: {repr(value[:100])}...")
            else:
                print(f"✓ {field}: {repr(value)}")
        else:
            print(f"✗ {field}: MISSING")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    

if __name__ == "__main__":
    test_pdf()
