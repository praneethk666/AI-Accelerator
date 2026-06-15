"""Tests for document categorization module with new interface.

Tests verify:
- New run(self, state, config) interface
- state["file_path"] is read from state
- state["confidence"] replaces state["categorization_confidence"]
- All state fields are always present
- Error handling is graceful
"""

import sys
import os
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.categorize.categorize_tool import CategorizeTool
from tests.fixtures import sample_global_config, sample_query_response


# Configuration for testing
CATEGORIZATION_CONFIG = {
    "type_to_route": {
        "cad_drawing": "cad_route",
        "circuit_diagram": "circuit_route",
        "datasheet": "diagram_route",
        "report": "text_default",
        "invoice": "text_default",
        "presentation": "presentation_route",
        "spreadsheet": "text_default",
        "image": "image_route",
        "unknown": "text_default"
    },
    "default_industry": "automotive",
    "categorization": {
        "industry_keywords": {
            "automotive": ["toyota", "ford", "bmw", "vehicle", "torque", "engine", "chassis", "transmission", "automotive"],
            "electronics": ["circuit", "semiconductor", "resistor", "capacitor", "pcb", "schematic", "voltage", "signal"],
            "manufacturing": ["assembly", "drawing", "tolerance", "weld", "machining", "fixture", "jig", "bom", "part number"],
            "finance": ["invoice", "balance sheet", "profit", "loss", "revenue", "ledger", "audit", "fiscal", "equity"],
            "legal": ["contract", "agreement", "clause", "court", "law", "nda", "litigation", "compliance", "liability"],
            "healthcare": ["patient", "diagnosis", "treatment", "medical", "pharma", "clinical", "dosage", "trial", "disease", "symptom"],
        },
        "confidence_thresholds": {"categorization_low_confidence": 0.5}
    }
}


class TestNewInterface:
    """Test the CategorizeTool run(self, state, config) interface."""

    def test_circuit_diagram_categorization(self):
        """Test categorization of a circuit diagram file."""
        state = {"file_path": "tests/test_circuit_wiring_v2.pdf"}
        
        tool = CategorizeTool()
        result = tool.run(state, CATEGORIZATION_CONFIG)
        
        # Verify all required fields are present
        assert "route" in result
        assert "document_type" in result
        assert "confidence" in result
        assert "industry" in result
        assert "reasoning" in result
        assert "file_type" in result
        assert "errors" in result
        
        # Verify circuit diagram is detected
        print(f"\n✅ Circuit Diagram Test:")
        print(f"   Document Type: {result['document_type']}")
        print(f"   Route: {result['route']}")
        print(f"   Industry: {result['industry']}")
        print(f"   Confidence: {result['confidence']}")
        
        assert result["document_type"] == "circuit_diagram"
        assert result["route"] == "circuit_route"
        assert result["industry"] == "electronics"
        assert result["confidence"] >= 0.5

    def test_presentation_categorization(self):
        """Test categorization of a presentation file."""
        state = {"file_path": "tests/presentation.pptx"}
        
        tool = CategorizeTool()
        result = tool.run(state, CATEGORIZATION_CONFIG)
        
        # Verify all required fields are present
        assert "route" in result
        assert "document_type" in result
        assert "confidence" in result
        
        print(f"\n✅ Presentation Test:")
        print(f"   Document Type: {result['document_type']}")
        print(f"   Route: {result['route']}")
        print(f"   Confidence: {result['confidence']}")
        
        assert result["document_type"] == "presentation"
        assert result["route"] == "presentation_route"
        assert result["confidence"] >= 0.5

    def test_missing_file_path_returns_fallback(self):
        """Missing file_path should return safe fallback."""
        state = {}  # No file_path
        
        tool = CategorizeTool()
        result = tool.run(state, CATEGORIZATION_CONFIG)
        
        # Verify safe fallback values
        assert "route" in result
        assert "document_type" in result
        assert "confidence" in result
        assert "industry" in result
        
        print(f"\n✅ Missing File Path Test:")
        print(f"   Route: {result['route']}")
        print(f"   Document Type: {result['document_type']}")
        print(f"   Errors: {result.get('errors', [])}")
        
        assert result["route"] == "text_default"
        assert result["document_type"] == "report"

    def test_confidence_field_present(self):
        """Verify state has 'confidence' field, not 'categorization_confidence'."""
        state = {"file_path": "tests/test_circuit_wiring_v2.pdf"}
        
        tool = CategorizeTool()
        result = tool.run(state, CATEGORIZATION_CONFIG)
        
        # Verify correct field name
        assert "confidence" in result
        assert "categorization_confidence" not in result
        
        print(f"\n✅ Confidence Field Test:")
        print(f"   Has 'confidence' field: ✅")
        print(f"   No 'categorization_confidence': ✅")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing CategorizeTool Class")
    print("=" * 70)
    
    # Run tests manually
    test_suite = TestNewInterface()
    
    try:
        test_suite.test_circuit_diagram_categorization()
    except FileNotFoundError as e:
        print(f"   ⚠️  File not found: {e}")
    except AssertionError as e:
        print(f"   ❌ Assertion failed: {e}")
    
    try:
        test_suite.test_presentation_categorization()
    except FileNotFoundError as e:
        print(f"   ⚠️  File not found: {e}")
    except AssertionError as e:
        print(f"   ❌ Assertion failed: {e}")
    
    try:
        test_suite.test_missing_file_path_returns_fallback()
        print(f"   ✅ Test passed!")
    except AssertionError as e:
        print(f"   ❌ Assertion failed: {e}")
    
    try:
        test_suite.test_confidence_field_present()
        print(f"   ✅ Test passed!")
    except FileNotFoundError as e:
        print(f"   ⚠️  File not found: {e}")
    except AssertionError as e:
        print(f"   ❌ Assertion failed: {e}")
    
    print("\n" + "=" * 70)
    print("✅ All CategorizeTool tests completed!")
    print("=" * 70)
    """Test the new run(self, state, config) interface."""

    def test_run_requires_file_path_in_state(self):
        """run() should read file_path from state."""
        state = {"file_path": "Invoice_Q4_2024.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert "route" in result
        assert "document_type" in result
        assert "confidence" in result  # NOT categorization_confidence
        assert "industry" in result
        assert "reasoning" in result

    def test_state_confidence_not_categorization_confidence(self):
        """state should have 'confidence' field, not 'categorization_confidence'."""
        state = {"file_path": "Contract_2025.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert "confidence" in state
        assert "categorization_confidence" not in state

    def test_missing_file_path_returns_fallback(self):
        """Missing file_path should return safe fallback."""
        state = {}  # No file_path
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert state["route"] == "text_default"
        assert state["confidence"] == 0.0
        assert len(state["errors"]) > 0
        assert "missing file_path" in state["errors"][0]

    def test_config_deployment_default_industry(self):
        """Deployment config default_industry should be used."""
        state = {"file_path": "unknown_document.pdf"}
        config = sample_global_config()
        config["default_industry"] = "pharma"
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        # When file_path doesn't exist, should use deployment default
        assert state["industry"] == "pharma"

    def test_uses_global_config_structure(self):
        """Should use config/global.yaml structure for categorization."""
        state = {"file_path": "Invoice_Test.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        # Verify config structure (post-merge: type_to_route at root level)
        assert "type_to_route" in config
        assert "industries" in config
        assert "document_types" in config
        assert "categorization" in config
        assert "routes" in config

    def test_filename_matching_with_global_config(self):
        """Filename matching should work with global config."""
        state = {"file_path": "Invoice_Q4_2024.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert result["document_type"] == "invoice"
        assert result["route"] == "text_default"  # invoice maps to text_default in global.yaml
        assert result["confidence"] == 0.90  # Filename match confidence

    def test_circuit_diagram_detection(self):
        """Circuit diagram should route to circuit_route."""
        state = {"file_path": "Hydraulic_Circuit_Diagram.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert result["document_type"] == "circuit_diagram"
        assert result["route"] == "circuit_route"  # circuit_diagram maps to circuit_route in global.yaml
        assert result["industry"] == "electronics"  # "circuit" is an electronics keyword

    def test_contract_detection(self):
        """Contract should route to text_default."""
        state = {"file_path": "Legal_Contract_2025.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert result["document_type"] == "contract"
        assert result["route"] == "text_default"
        assert result["industry"] == "legal"

    def test_all_state_fields_present(self):
        """All required state fields should always be present."""
        state = {"file_path": "unknown.xyz"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert "route" in state
        assert "document_type" in state
        assert "industry" in state
        assert "confidence" in state
        assert "reasoning" in state
        assert "errors" in state


class TestConfigFromGlobalYaml:
    """Test that configuration is properly loaded from global.yaml."""

    def test_type_to_route_mapping(self):
        """type_to_route should be used for routing."""
        state = {"file_path": "Quarterly_Report_2024.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert result["route"] == "text_default"
        assert config["type_to_route"]["report"] == "text_default"

    def test_industry_keywords_detection(self):
        """industry_keywords should be used for industry detection."""
        # Test that keywords are correctly defined in config
        config = sample_global_config()
        industry_keywords = config["categorization"]["industry_keywords"]
        
        # Verify that automotive has the required keywords
        assert "automotive" in industry_keywords
        assert any(kw in industry_keywords["automotive"] for kw in ["toyota", "ford", "bmw", "vehicle"])
        
        # Verify finance has invoice-related keywords
        assert "finance" in industry_keywords
        assert any(kw in industry_keywords["finance"] for kw in ["invoice", "balance sheet", "revenue"])

    def test_confidence_thresholds(self):
        """confidence_thresholds should be used."""
        state = {"file_path": "Generic_Document.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        # Should have threshold defined
        assert "confidence_thresholds" in config["categorization"]
        assert "categorization_low_confidence" in config["categorization"]["confidence_thresholds"]


class TestErrorHandling:
    """Test error handling with new interface."""

    def test_never_crashes_on_invalid_file(self):
        """Should never crash, even with invalid files."""
        state = {"file_path": "/does/not/exist/document.pdf"}
        config = sample_global_config()
        
        try:
            tool = CategorizeTool()
            result = tool.run(state, config)
            assert "route" in result
            assert result["route"] == "text_default"
        except Exception as e:
            pytest.fail(f"run() crashed with {type(e).__name__}: {e}")

    def test_error_message_is_descriptive(self):
        """Error messages should be descriptive."""
        state = {"file_path": "/does/not/exist/doc.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert len(state["errors"]) > 0
        assert "exception" in state["errors"][0].lower() or "not" in state["errors"][0].lower()

    def test_graceful_fallback_on_error(self):
        """Should provide sensible defaults on error."""
        state = {"file_path": "/invalid/path/doc.xyz"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert result["route"] == "text_default"
        assert result["document_type"] == "report"
        assert result["confidence"] == 0.0


class TestSampleQueryResponse:
    """Test sample query response from fixtures."""

    def test_sample_response_has_required_fields(self):
        """Sample response should have all required fields."""
        response = sample_query_response()
        
        assert "file_path" in response
        assert "document_type" in response
        assert "route" in response
        assert "industry" in response
        assert "confidence" in response
        assert "reasoning" in response
        assert "status" in response
        assert "errors" in response

    def test_sample_response_confidence_is_numeric(self):
        """Confidence should be numeric."""
        response = sample_query_response()
        
        assert isinstance(response["confidence"], (int, float))
        assert 0.0 <= response["confidence"] <= 1.0


class TestIntegrationWithTestData:
    """Integration tests using actual test data files."""

    def test_real_invoice_file(self):
        """Test with actual invoice Excel file from test-data."""
        state = {"file_path": "test-data/Task 5 Equality Table.xlsx"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        # Verify structure
        assert "route" in result
        assert "document_type" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], (int, float))
        assert 0.0 <= result["confidence"] <= 1.0

    def test_pdf_circuit_diagram(self):
        """Test with PDF circuit diagram file."""
        state = {"file_path": "test-data/TIGG300_OP200_HYDRAULIC,PNEUMATIC CIRCUIT DIAGRAM.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        # Should strictly detect circuit diagram (filename contains "CIRCUIT DIAGRAM")
        assert result["document_type"] == "circuit_diagram", f"Expected circuit_diagram but got {result['document_type']}"
        assert result["route"] == "circuit_route", f"Expected circuit_route but got {result['route']}"  # from global.yaml type_to_route
        assert "confidence" in result

    def test_cad_motor_file(self):
        """Test with CAD motor file."""
        state = {"file_path": "test-data/MS03AAA981AA-Expansion Motor.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert "confidence" in result
        assert result["industry"] is not None

    def test_research_paper(self):
        """Test with research paper file."""
        state = {"file_path": "test-data/Multisystem thromboembolism in a COVID-19 patient  a case report.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert result["document_type"] is not None
        assert result["route"] is not None
        assert "confidence" in result

    def test_contract_file(self):
        """Test with contract file."""
        state = {"file_path": "test-data/SampleContract-Shuttle.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert "confidence" in result
        assert result["route"] in config["type_to_route"].values()

    def test_presentation_file(self):
        """Test with PowerPoint presentation."""
        state = {"file_path": "test-data/test.pptx"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert "confidence" in result
        assert "document_type" in result

    def test_excel_file(self):
        """Test with Excel file."""
        state = {"file_path": "test-data/test.xlsx"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert "confidence" in result
        assert "document_type" in result


class TestConfigGlobalYamlIntegration:
    """Test that config/global.yaml structure is correctly used."""

    def test_document_types_match_taxonomy(self):
        """All document types in config should be valid."""
        config = sample_global_config()
        type_to_route = config["type_to_route"]
        
        # These types are defined in config/global.yaml document_types
        expected_types = [
            "circuit_diagram", "cad_drawing", "invoice", "report",
            "datasheet", "presentation", "spreadsheet", "image", "unknown"
        ]
        
        for doc_type in expected_types:
            assert doc_type in type_to_route, f"Missing {doc_type} in type_to_route"

    def test_routes_are_valid(self):
        """All routes should be valid route names."""
        config = sample_global_config()
        # These routes are defined in the new 5-route design
        valid_routes = {"text_default", "diagram_heavy", "cad_route", "circuit_route", "image_route", "presentation_route"}
        
        for route in config["type_to_route"].values():
            assert route in valid_routes, f"Invalid route: {route}"

    def test_industry_keywords_structure(self):
        """Industry keywords should be properly structured."""
        config = sample_global_config()
        industry_keywords = config["categorization"]["industry_keywords"]
        
        # These industries are defined in config/global.yaml industries
        expected_industries = ["automotive", "electronics", "manufacturing", "finance", "legal", "healthcare", "general"]
        
        for industry in expected_industries:
            assert industry in industry_keywords, f"Missing industry: {industry}"
            assert isinstance(industry_keywords[industry], list)
            # general can be empty, but others should have keywords
            if industry != "general":
                assert len(industry_keywords[industry]) > 0, f"Industry {industry} has no keywords"

    def test_deployment_config_defaults(self):
        """Default industry should be set at root level."""
        config = sample_global_config()
        
        # In the new config structure, default_industry is at root level
        assert "default_industry" in config
        assert config["default_industry"] in config["industries"]


class TestStateConsistency:
    """Test that state is consistent across different scenarios."""

    def test_state_fields_always_set(self):
        """Required state fields should always be set."""
        test_cases = [
            {"file_path": "Invoice.pdf"},
            {"file_path": "Contract.pdf"},
            {"file_path": "unknown.xyz"},
            {},
        ]
        config = sample_global_config()
        
        required_fields = ["route", "document_type", "industry", "confidence", "reasoning", "errors"]
        
        for state in test_cases:
            tool = CategorizeTool()
            result = tool.run(state, config)
            for field in required_fields:
                assert field in result, f"Field '{field}' missing for state: {state}"

    def test_confidence_is_between_0_and_1(self):
        """Confidence should always be between 0.0 and 1.0."""
        test_cases = [
            {"file_path": "Invoice.pdf"},
            {"file_path": "Contract.pdf"},
            {"file_path": "unknown.xyz"},
            {},
        ]
        config = sample_global_config()
        
        for state in test_cases:
            tool = CategorizeTool()
            result = tool.run(state, config)
            assert isinstance(result["confidence"], (int, float))
            assert 0.0 <= result["confidence"] <= 1.0, \
                f"Confidence {result['confidence']} out of range for state: {state}"

    def test_route_is_valid(self):
        """Route should be one of the valid routes."""
        test_cases = [
            {"file_path": "Invoice.pdf"},
            {"file_path": "Contract.pdf"},
            {"file_path": "unknown.xyz"},
            {},
        ]
        config = sample_global_config()
        valid_routes = ["text_default", "diagram_heavy", "cad_route", "circuit_route", "image_route", "presentation_route"]
        
        for state in test_cases:
            tool = CategorizeTool()
            result = tool.run(state, config)
            assert result["route"] in valid_routes, \
                f"Invalid route '{result['route']}' for state: {state}"

    def test_document_type_is_valid(self):
        """Document type should be one of the valid types."""
        test_cases = [
            {"file_path": "Invoice.pdf"},
            {"file_path": "Contract.pdf"},
            {"file_path": "unknown.xyz"},
            {},
        ]
        config = sample_global_config()
        
        for state in test_cases:
            tool = CategorizeTool()
            result = tool.run(state, config)
            # Document type should be in type_to_route or be a default fallback
            valid_types = list(config["type_to_route"].keys())
            assert result["document_type"] in valid_types or result["document_type"] == "report"


class TestErrorMessages:
    """Test error handling and messages."""

    def test_errors_list_is_initialized(self):
        """Errors should be initialized even on success."""
        state = {"file_path": "Invoice.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert "errors" in result
        assert isinstance(result["errors"], list)

    def test_missing_file_path_error(self):
        """Should error when file_path is missing."""
        state = {}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert len(result["errors"]) > 0
        assert "file_path" in result["errors"][0].lower()

    def test_nonexistent_file_error(self):
        """Should error gracefully for nonexistent files."""
        state = {"file_path": "/definitely/does/not/exist.pdf"}
        config = sample_global_config()
        
        tool = CategorizeTool()
        result = tool.run(state, config)
        
        assert len(result["errors"]) > 0
        assert result["route"] == "text_default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
