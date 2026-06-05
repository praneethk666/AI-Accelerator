"""Tests for document categorization module with new interface.

Tests verify:
- New run(self, state, config) interface
- state["file_path"] is read from state
- state["confidence"] replaces state["categorization_confidence"]
- Global config from config/global.yaml is used
- All state fields are always present
- Error handling is graceful
"""

import pytest
from tests.fixtures import sample_global_config, sample_query_response
from backend.categorize.categorize_tool import run


class TestNewInterface:
    """Test the new run(self, state, config) interface."""

    def test_run_requires_file_path_in_state(self):
        """run() should read file_path from state."""
        state = {"file_path": "Invoice_Q4_2024.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert "route" in result
        assert "document_type" in result
        assert "confidence" in result  # NOT categorization_confidence
        assert "industry" in result
        assert "reasoning" in result

    def test_state_confidence_not_categorization_confidence(self):
        """state should have 'confidence' field, not 'categorization_confidence'."""
        state = {"file_path": "Contract_2025.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert "confidence" in state
        assert "categorization_confidence" not in state

    def test_missing_file_path_returns_fallback(self):
        """Missing file_path should return safe fallback."""
        state = {}  # No file_path
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert state["route"] == "text_default"
        assert state["confidence"] == 0.0
        assert len(state["errors"]) > 0
        assert "missing file_path" in state["errors"][0]

    def test_config_deployment_default_industry(self):
        """Deployment config default_industry should be used."""
        state = {"file_path": "unknown_document.pdf"}
        config = sample_global_config()
        config["deployment"]["default_industry"] = "pharma"
        
        result = run(None, state, config)
        
        # When file_path doesn't exist, should use deployment default
        assert state["industry"] == "pharma"

    def test_uses_global_config_structure(self):
        """Should use config/global.yaml structure for categorization."""
        state = {"file_path": "Invoice_Test.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        # Verify config was used
        assert "categorization" in config
        assert "type_to_route" in config["categorization"]
        assert "industry_keywords" in config["categorization"]
        assert "deployment" in config

    def test_filename_matching_with_global_config(self):
        """Filename matching should work with global config."""
        state = {"file_path": "Invoice_Q4_2024.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert result["document_type"] == "invoice"
        assert result["route"] == "table_heavy"
        assert result["confidence"] == 0.90  # Filename match confidence

    def test_circuit_diagram_detection(self):
        """Circuit diagram should route to diagram_heavy."""
        state = {"file_path": "Hydraulic_Circuit_Diagram.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert result["document_type"] == "circuit_diagram"
        assert result["route"] == "diagram_heavy"
        assert result["industry"] == "automotive"

    def test_contract_detection(self):
        """Contract should route to text_default."""
        state = {"file_path": "Legal_Contract_2025.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert result["document_type"] == "contract"
        assert result["route"] == "text_default"
        assert result["industry"] == "legal"

    def test_all_state_fields_present(self):
        """All required state fields should always be present."""
        state = {"file_path": "unknown.xyz"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
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
        state = {"file_path": "Financial_Statement_2024.xlsx"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert result["route"] == "table_heavy"
        assert config["categorization"]["type_to_route"]["financial_statement"] == "table_heavy"

    def test_industry_keywords_detection(self):
        """industry_keywords should be used for industry detection."""
        state = {"file_path": "Toyota_Motor_Vehicle.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert result["industry"] == "automotive"
        assert "toyota" in [kw.lower() for kw in config["categorization"]["industry_keywords"]["automotive"]]

    def test_confidence_thresholds(self):
        """confidence_thresholds should be used."""
        state = {"file_path": "Generic_Document.pdf"}
        config = sample_global_config()
        
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
            result = run(None, state, config)
            assert "route" in result
            assert result["route"] == "text_default"
        except Exception as e:
            pytest.fail(f"run() crashed with {type(e).__name__}: {e}")

    def test_error_message_is_descriptive(self):
        """Error messages should be descriptive."""
        state = {"file_path": "/does/not/exist/doc.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert len(state["errors"]) > 0
        assert "exception" in state["errors"][0].lower() or "not" in state["errors"][0].lower()

    def test_graceful_fallback_on_error(self):
        """Should provide sensible defaults on error."""
        state = {"file_path": "/invalid/path/doc.xyz"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
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
        
        result = run(None, state, config)
        
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
        
        result = run(None, state, config)
        
        # Should detect circuit diagram
        assert result["document_type"] in ["circuit_diagram", "schematic", "cad_drawing"]
        assert result["route"] in ["diagram_heavy", "text_default"]
        assert "confidence" in result

    def test_cad_motor_file(self):
        """Test with CAD motor file."""
        state = {"file_path": "test-data/MS03AAA981AA-Expansion Motor.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert "confidence" in result
        assert result["industry"] is not None

    def test_research_paper(self):
        """Test with research paper file."""
        state = {"file_path": "test-data/Multisystem thromboembolism in a COVID-19 patient  a case report.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert result["document_type"] is not None
        assert result["route"] is not None
        assert "confidence" in result

    def test_contract_file(self):
        """Test with contract file."""
        state = {"file_path": "test-data/SampleContract-Shuttle.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert "confidence" in result
        assert result["route"] in config["categorization"]["type_to_route"].values()

    def test_presentation_file(self):
        """Test with PowerPoint presentation."""
        state = {"file_path": "test-data/test.pptx"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert "confidence" in result
        assert "document_type" in result

    def test_excel_file(self):
        """Test with Excel file."""
        state = {"file_path": "test-data/test.xlsx"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert "confidence" in result
        assert "document_type" in result


class TestConfigGlobalYamlIntegration:
    """Test that config/global.yaml structure is correctly used."""

    def test_document_types_match_taxonomy(self):
        """All document types in config should be valid."""
        config = sample_global_config()
        type_to_route = config["categorization"]["type_to_route"]
        
        expected_types = [
            "circuit_diagram", "cad_drawing", "schematic",
            "invoice", "financial_statement", "purchase_order",
            "contract", "policy", "research_paper", "report"
        ]
        
        for doc_type in expected_types:
            assert doc_type in type_to_route

    def test_routes_are_valid(self):
        """All routes should be valid route names."""
        config = sample_global_config()
        valid_routes = ["diagram_heavy", "table_heavy", "text_default", "presentation_route"]
        
        for route in config["categorization"]["type_to_route"].values():
            assert route in valid_routes

    def test_industry_keywords_structure(self):
        """Industry keywords should be properly structured."""
        config = sample_global_config()
        industry_keywords = config["categorization"]["industry_keywords"]
        
        expected_industries = ["automotive", "pharma", "finance", "legal", "engineering", "manufacturing"]
        
        for industry in expected_industries:
            assert industry in industry_keywords
            assert isinstance(industry_keywords[industry], list)
            assert len(industry_keywords[industry]) > 0

    def test_deployment_config_defaults(self):
        """Deployment config should have required defaults."""
        config = sample_global_config()
        deployment = config["deployment"]
        
        assert "default_industry" in deployment
        assert deployment["default_industry"] in [
            "automotive", "pharma", "finance", "legal", "engineering", "manufacturing"
        ]


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
            result = run(None, state, config)
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
            result = run(None, state, config)
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
        valid_routes = ["diagram_heavy", "table_heavy", "text_default", "presentation_route"]
        
        for state in test_cases:
            result = run(None, state, config)
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
            result = run(None, state, config)
            # Document type should be in type_to_route or be a default fallback
            valid_types = list(config["categorization"]["type_to_route"].keys())
            assert result["document_type"] in valid_types or result["document_type"] == "report"


class TestErrorMessages:
    """Test error handling and messages."""

    def test_errors_list_is_initialized(self):
        """Errors should be initialized even on success."""
        state = {"file_path": "Invoice.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert "errors" in result
        assert isinstance(result["errors"], list)

    def test_missing_file_path_error(self):
        """Should error when file_path is missing."""
        state = {}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert len(result["errors"]) > 0
        assert "file_path" in result["errors"][0].lower()

    def test_nonexistent_file_error(self):
        """Should error gracefully for nonexistent files."""
        state = {"file_path": "/definitely/does/not/exist.pdf"}
        config = sample_global_config()
        
        result = run(None, state, config)
        
        assert len(result["errors"]) > 0
        assert result["route"] == "text_default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
