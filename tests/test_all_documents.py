#!/usr/bin/env python
"""Test categorization on all test documents - Output as JSON with file_type."""
import os
import sys
import json
from collections import defaultdict

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.categorize.categorize_tool import CategorizeTool
from tests.fixtures import sample_global_config

test_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test-data')
files = sorted([f for f in os.listdir(test_data_dir) if f != 'README.md'])

tool = CategorizeTool()
config = sample_global_config()

results = []
results_by_route = defaultdict(list)

print("\n" + "="*180)
print(f"DOCUMENT CATEGORIZATION RESULTS - {len(files)} files")
print("="*180)

for filename in files:
    file_path = os.path.join(test_data_dir, filename)
    try:
        result = tool.run({'file_path': file_path}, config)
        route = result.get('route', 'N/A')
        doc_type = result.get('document_type', 'N/A')
        file_type = result.get('file_type', 'unknown')
        industry = result.get('industry', 'N/A')
        confidence = result.get('confidence', 0)
        
        # Track by route
        results_by_route[route].append(filename)
        
        results.append({
            'filename': filename,
            'route': route,
            'document_type': doc_type,
            'file_type': file_type,
            'industry': industry,
            'confidence': confidence,
            'reasoning': result.get('reasoning', '')
        })
        
    except Exception as e:
        results.append({
            'filename': filename,
            'route': 'ERROR',
            'document_type': 'ERROR',
            'file_type': 'unknown',
            'industry': 'N/A',
            'confidence': 0,
            'reasoning': str(e)[:100]
        })

# Print table
print(f"{'Filename':<50} | {'Route':<18} | {'File Type':<12} | {'Type':<18} | {'Industry':<12} | {'Conf':<6}")
print("-"*180)
for r in results:
    print(f"{r['filename']:<50} | {r['route']:<18} | {r['file_type']:<12} | {r['document_type']:<18} | {r['industry']:<12} | {str(r['confidence']):<6}")

print("="*180)

# Print JSON format
print("\n[FULL JSON OUTPUT]\n")
json_output = {
    'summary': {
        'total_documents': len(results),
        'total_routes': len(results_by_route),
        'route_distribution': {route: len(docs) for route, docs in sorted(results_by_route.items())}
    },
    'results': results
}

print(json.dumps(json_output, indent=2))

# Print summary by route
print("\n" + "="*180)
print("SUMMARY BY ROUTE:")
print("="*180)
for route in sorted(results_by_route.keys()):
    count = len(results_by_route[route])
    print(f"\n{route:<20} : {count:>3} documents")
    for filename in sorted(results_by_route[route]):
        doc = next((r for r in results if r['filename'] == filename), None)
        if doc:
            print(f"  - {doc['filename']:<50} ({doc['file_type']:<12}) confidence: {doc['confidence']:.2f}")

print("\n" + "="*180)
print("DOCUMENT COUNT BY FILE TYPE:")
print("="*180)
file_type_count = defaultdict(int)
for r in results:
    file_type_count[r['file_type']] += 1

for ft in sorted(file_type_count.keys()):
    count = file_type_count[ft]
    print(f"{ft:<15}: {count} documents")
