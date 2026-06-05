#!/usr/bin/env python
"""
CAD Document Detection Verification Script
Verifies that the CAD fix is working correctly
"""

from backend.categorize.categorize_tool import run
from backend.categorize.text_extractor import is_cad_document_by_filename
import json
from pathlib import Path


def test_cad_filename_detection():
    """Test CAD filename pattern detection"""
    print("\n" + "="*70)
    print("TEST 1: CAD Filename Pattern Detection")
    print("="*70)
    
    test_cases = [
        ("MS03AAA981AA-Expansion Motor.pdf", True),
        ("DWG-2024-001-Assembly.pdf", True),
        ("drawing_schematic_v3.pdf", True),
        ("Motor Assembly Drawing.pdf", True),
        ("engine_bracket_r2.pdf", True),
        ("report_2024_Q1.pdf", False),
        ("contract.pdf", False),
        ("invoice.pdf", False),
    ]
    
    passed = 0
    for filename, expected in test_cases:
        result = is_cad_document_by_filename(filename)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"  {status}: {filename} → {result} (expected {expected})")
        if result == expected:
            passed += 1
    
    print(f"\nResult: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_motor_drawing_classification():
    """Test the critical motor drawing classification"""
    print("\n" + "="*70)
    print("TEST 2: Motor Drawing Classification (Critical)")
    print("="*70)
    
    motor_file = r"D:\AI-Accelerator\MS03AAA981AA-Expansion Motor.pdf"
    
    if not Path(motor_file).exists():
        print(f"  ⚠️  File not found: {motor_file}")
        return False
    
    print(f"  Testing: {Path(motor_file).name}")
    
    state = {}
    result = run(motor_file, state)
    
    # Check critical fields
    checks = [
        ("document_type", "cad_drawing", result["document_type"]),
        ("route", "diagram_heavy", result["route"]),
        ("industry", "automotive", result["industry"]),
        ("confidence >= 0.75", True, result["confidence"] >= 0.75),
        ("errors empty", True, len(state.get("errors", [])) == 0),
    ]
    
    all_passed = True
    for check_name, expected, actual in checks:
        passed = expected == actual if expected is not True else actual
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}: {actual} (expected {expected})")
        all_passed = all_passed and passed
    
    print(f"\n  Classification Result:")
    print(f"    Type: {result['document_type']}")
    print(f"    Route: {result['route']}")
    print(f"    Industry: {result['industry']}")
    print(f"    Confidence: {result['confidence']:.2f}")
    
    return all_passed


def test_other_document_types():
    """Verify non-CAD documents still work"""
    print("\n" + "="*70)
    print("TEST 3: Non-CAD Document Types")
    print("="*70)
    
    test_files = [
        (r"D:\AI-Accelerator\test.pptx", "presentation", "presentation_route"),
    ]
    
    all_passed = True
    for file_path, expected_type, expected_route in test_files:
        if not Path(file_path).exists():
            print(f"  ⚠️  File not found: {file_path}")
            continue
        
        print(f"  Testing: {Path(file_path).name}")
        
        state = {}
        result = run(file_path, state)
        
        type_match = result["document_type"] == expected_type
        route_match = result["route"] == expected_route
        
        type_status = "✅ PASS" if type_match else "❌ FAIL"
        route_status = "✅ PASS" if route_match else "❌ FAIL"
        
        print(f"    {type_status}: Type {result['document_type']} (expected {expected_type})")
        print(f"    {route_status}: Route {result['route']} (expected {expected_route})")
        
        all_passed = all_passed and type_match and route_match
    
    return all_passed


def test_confidence_scoring():
    """Verify confidence scoring is reasonable"""
    print("\n" + "="*70)
    print("TEST 4: Confidence Scoring")
    print("="*70)
    
    motor_file = r"D:\AI-Accelerator\MS03AAA981AA-Expansion Motor.pdf"
    ppt_file = r"D:\AI-Accelerator\test.pptx"
    
    results = []
    
    for file_path, expected_min, expected_max in [
        (motor_file, 0.7, 1.0),
        (ppt_file, 0.85, 1.0),
    ]:
        if not Path(file_path).exists():
            continue
        
        state = {}
        result = run(file_path, state)
        conf = result["confidence"]
        
        in_range = expected_min <= conf <= expected_max
        status = "✅ PASS" if in_range else "❌ FAIL"
        print(f"  {status}: {Path(file_path).name} confidence {conf:.2f} (expected {expected_min}-{expected_max})")
        
        results.append(in_range)
    
    return all(results)


def print_summary(tests_passed):
    """Print test summary"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    test_names = [
        "CAD Filename Detection",
        "Motor Drawing Classification (Critical)",
        "Non-CAD Document Types",
        "Confidence Scoring",
    ]
    
    for i, (name, passed) in enumerate(zip(test_names, tests_passed)):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} Test {i+1}: {name}")
    
    total_passed = sum(tests_passed)
    total_tests = len(tests_passed)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! CAD detection is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. See details above.")
        return False


if __name__ == "__main__":
    print("\n🔍 CAD Document Detection Verification")
    print("="*70)
    
    tests = [
        test_cad_filename_detection(),
        test_motor_drawing_classification(),
        test_other_document_types(),
        test_confidence_scoring(),
    ]
    
    success = print_summary(tests)
    exit(0 if success else 1)
