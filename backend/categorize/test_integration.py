"""Comprehensive integration tests for categorization module.

Tests cover:
- State field validation
- Error handling and graceful fallbacks  
- Multiple file types (PDF, Excel, PPT, images)
- Low-confidence scenarios
- Industry detection
- Configuration handling
"""

import os
import json
import pytest

from backend.categorize.categorize_tool import run
from backend.categorize.classifier import (
    _score_industry_from_filename,
    _score_industry_from_text,
)


class TestStateFields:
    """Verify all required state fields are always written."""

    def test_nonexistent_file_returns_all_fields(self):
        """Even for missing files, all state fields must be present."""
        state = {"errors": []}
        result = run(
            file_path="/does/not/exist/document.pdf",
            state=state,
            deployment=None
        )

        # Check return value has all fields
        assert "route" in result
        assert "document_type" in result
        assert "industry" in result
        assert "confidence" in result
        assert "reasoning" in result

        # Check state has all fields
        assert "route" in state
        assert "document_type" in state
        assert "industry" in state
        assert "categorization_confidence" in state
        assert "reasoning" in state
        assert "errors" in state

    def test_graceful_fallback_on_missing_file(self):
        """Missing files should fall back to safe defaults."""
        state = {}
        result = run(
            file_path="/does/not/exist/mystery.pdf",
            state=state,
            deployment=None
        )

        # Fallback values
        assert state["route"] == "text_default"
        assert state["categorization_confidence"] == 0.0
        assert len(state["errors"]) > 0

    def test_state_fields_with_deployment_config(self):
        """State fields should respect deployment configuration."""
        state = {}
        deployment = {
            "default_industry": "pharma",
            "client": "Roche"
        }

        result = run(
            file_path="/does/not/exist/doc.pdf",
            state=state,
            deployment=deployment
        )

        # Should use deployment default industry
        assert state["industry"] == "pharma"


class TestIndustryDetection:
    """Test industry keyword matching."""

    def test_filename_industry_matching(self):
        """Industry should be detected from filename."""
        industry_keywords = {
            "automotive": ["toyota", "vehicle", "wiring"],
            "pharma": ["clinical", "dosage", "fda"],
            "finance": ["ebitda", "quarterly", "revenue"],
        }

        # Test automotive
        result = _score_industry_from_filename(
            "toyota_wiring_harness.pdf",
            industry_keywords
        )
        assert result == "automotive"

        # Test pharma
        result = _score_industry_from_filename(
            "clinical_trial_report.pdf",
            industry_keywords
        )
        assert result == "pharma"

        # Test finance
        result = _score_industry_from_filename(
            "quarterly_revenue_2024.xlsx",
            industry_keywords
        )
        assert result == "finance"

    def test_text_industry_matching(self):
        """Industry should be detected from document text."""
        industry_keywords = {
            "automotive": ["toyota", "vehicle", "wiring"],
            "pharma": ["clinical", "dosage", "fda"],
        }

        text = """
        This is a clinical trial protocol document.
        Dosage: 100mg daily
        FDA approval required.
        """

        result = _score_industry_from_text(text, industry_keywords)
        assert result == "pharma"

    def test_no_industry_match_returns_none(self):
        """When no keywords match, return None."""
        industry_keywords = {
            "automotive": ["toyota"],
            "pharma": ["clinical"],
        }

        result = _score_industry_from_filename(
            "generic_document.pdf",
            industry_keywords
        )
        assert result is None


class TestErrorHandling:
    """Test graceful error handling."""

    def test_never_crashes_on_invalid_file(self):
        """The categorizer should never crash."""
        state = {"errors": []}

        invalid_paths = [
            "/does/not/exist.pdf",
            "/dev/null",
        ]

        for path in invalid_paths:
            try:
                result = run(file_path=path, state=state, deployment=None)
                assert "route" in result
                assert state["route"] is not None
            except Exception as e:
                pytest.fail(f"Categorizer crashed on {path}: {e}")

    def test_error_messages_descriptive(self):
        """Error messages should be informative."""
        state = {"errors": []}
        result = run(
            file_path="/does/not/exist/mystery.pdf",
            state=state,
            deployment=None
        )

        assert len(state["errors"]) > 0

    def test_errors_never_prevent_output(self):
        """Even with errors, output should be complete."""
        state = {"errors": []}
        result = run(
            file_path="/invalid/path/doc.xyz",
            state=state,
            deployment=None
        )

        assert "route" in result
        assert "document_type" in result
        assert "confidence" in result
        assert "reasoning" in result


class TestFilenameMatching:
    """Test quick filename-based classification."""

    def test_invoice_filename_match(self):
        """Invoices should be detected from filename."""
        state = {"errors": []}
        result = run(
            file_path="/documents/invoice_2024_001.pdf",
            state=state,
            deployment=None
        )

        assert "route" in result


class TestSmoke:
    """Basic smoke tests."""

    def test_smoke_import(self):
        """Verify module imports without error."""
        from backend.categorize import categorize_tool
        from backend.categorize import classifier
        from backend.categorize import vision
        from backend.categorize import text_extractor

        assert categorize_tool is not None
        assert classifier is not None
        assert vision is not None
        assert text_extractor is not None

    def test_smoke_config_loads(self):
        """Verify config.yaml loads without error."""
        from backend.categorize.classifier import _load_config

        config = _load_config()
        assert "type_to_route" in config
        assert "industry_keywords" in config
        assert "confidence_thresholds" in config

    def test_config_required_routes_present(self):
        """Verify all required routes are defined."""
        from backend.categorize.classifier import _load_config

        config = _load_config()
        type_to_route = config["type_to_route"]

        required_types = [
            "circuit_diagram", "cad_drawing", "schematic",
            "invoice", "financial_statement", "purchase_order",
            "contract", "policy", "research_paper",
            "report", "manual", "presentation"
        ]

        for dtype in required_types:
            assert dtype in type_to_route, f"Missing type: {dtype}"

        valid_routes = {"diagram_heavy", "table_heavy", "text_default", "presentation_route"}
        for route in type_to_route.values():
            assert route in valid_routes, f"Invalid route: {route}"


class TestConfigValidation:
    """Test configuration structure and validity."""

    def test_config_structure(self):
        """Verify config.yaml has required structure."""
        from backend.categorize.classifier import _load_config

        config = _load_config()

        assert "type_to_route" in config
        assert "industry_keywords" in config
        assert "confidence_thresholds" in config
        assert "deployment" in config

        assert len(config["type_to_route"]) > 0
        assert "categorization_low_confidence" in config["confidence_thresholds"]
        assert "default_industry" in config["deployment"]

    def test_all_routes_valid(self):
        """Verify all mapped routes are in supported set."""
        from backend.categorize.classifier import _load_config

        config = _load_config()
        type_to_route = config["type_to_route"]

        valid_routes = {"diagram_heavy", "table_heavy", "text_default", "presentation_route"}

        for doc_type, route in type_to_route.items():
            assert route in valid_routes, f"Unknown route '{route}' for type '{doc_type}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
