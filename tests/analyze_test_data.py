"""Analyze all documents in test-data folder."""
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures import sample_global_config
from backend.categorize.categorize_tool import run

# Get all files from test-data
test_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test-data')
files = [f for f in os.listdir(test_data_dir) if os.path.isfile(os.path.join(test_data_dir, f)) and not f.startswith('.')]

config = sample_global_config()

print('=' * 110)
print('DOCUMENT CATEGORIZATION RESULTS - Test Data Files')
print('=' * 110)
print()

for i, filename in enumerate(sorted(files), 1):
    file_path = os.path.join(test_data_dir, filename)
    state = {'file_path': file_path}
    
    try:
        result = run(None, state, config)
        
        print(f"{i:2d}. FILE: {filename}")
        print(f"    Document Type: {result['document_type']}")
        print(f"    Route:         {result['route']}")
        print(f"    Industry:      {result['industry']}")
        print(f"    Confidence:    {result['confidence']:.2f}")
        reasoning = result['reasoning']
        if len(reasoning) > 100:
            print(f"    Reasoning:     {reasoning[:100]}...")
        else:
            print(f"    Reasoning:     {reasoning}")
        
        if result.get('errors'):
            print(f"    Errors:        {result['errors']}")
        print()
    except Exception as e:
        print(f"{i:2d}. FILE: {filename}")
        print(f"    ERROR: {str(e)[:80]}")
        print()

print('=' * 110)
