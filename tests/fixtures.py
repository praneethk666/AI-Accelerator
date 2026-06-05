"""Mock data fixtures for testing categorization module."""

from typing import Any, Dict


def sample_categorization_config() -> Dict[str, Any]:
    """Return a sample categorization configuration matching global.yaml structure."""
    return {
        "type_to_route": {
            "circuit_diagram": "diagram_heavy",
            "cad_drawing": "diagram_heavy",
            "schematic": "diagram_heavy",
            "invoice": "table_heavy",
            "financial_statement": "table_heavy",
            "purchase_order": "table_heavy",
            "spreadsheet": "table_heavy",
            "budget": "table_heavy",
            "forecast": "table_heavy",
            "report": "table_heavy",
            "contract": "text_default",
            "policy": "text_default",
            "research_paper": "text_default",
            "manual": "text_default",
            "presentation": "presentation_route",
        },
        "industry_keywords": {
            "automotive": [
                "toyota", "vehicle", "wiring", "harness", "chassis", "ecu",
                "motor", "engine", "transmission", "drive", "wheel"
            ],
            "pharma": [
                "clinical trial", "dosage", "fda", "adverse event", "pharmaceutical"
            ],
            "finance": [
                "ebitda", "balance sheet", "quarterly", "revenue", "invoice", "expense"
            ],
            "legal": [
                "agreement", "contract", "jurisdiction", "terms", "conditions"
            ],
            "engineering": [
                "voltage", "resistor", "schematic", "pcb", "bearing", "drawing"
            ],
            "manufacturing": [
                "production", "assembly", "tolerance", "qc", "iso", "bom", "machine"
            ],
        },
        "excel_document_types": {
            "invoice": ["invoice", "bill to", "amount due", "total"],
            "financial_statement": ["balance sheet", "income statement", "assets"],
            "purchase_order": ["purchase order", "po", "vendor", "quantity"],
            "budget": ["budget", "variance", "monthly"],
            "forecast": ["forecast", "projected", "estimate"],
            "spreadsheet": ["data", "calculation", "analysis"],
        },
        "confidence_thresholds": {
            "categorization_low_confidence": 0.5
        }
    }


def sample_global_config() -> Dict[str, Any]:
    """Return a complete global config matching config/global.yaml structure."""
    return {
        "categorization": sample_categorization_config(),
        "deployment": {
            "default_industry": "automotive",
            "client": "toyota",
        }
    }


def sample_query_response() -> Dict[str, Any]:
    """Return a sample document categorization response for testing frontend."""
    return {
        "file_path": "Invoice_Q4_2024.pdf",
        "document_type": "invoice",
        "route": "table_heavy",
        "industry": "finance",
        "confidence": 0.90,
        "reasoning": "Filename matched document_type='invoice'.",
        "status": "success",
        "errors": []
    }


def sample_categorization_results() -> Dict[str, Any]:
    """Return sample categorization results for multiple document types."""
    return {
        "total": 15,
        "successful": 11,
        "results": [
            {
                "label": "Invoice",
                "file": "Invoice_Q4_2024.pdf",
                "type": "invoice",
                "route": "table_heavy",
                "confidence": 0.90,
                "industry": "finance",
                "success": True,
            },
            {
                "label": "CAD Drawing",
                "file": "Motor_CAD_Drawing.pdf",
                "type": "cad_drawing",
                "route": "diagram_heavy",
                "confidence": 0.75,
                "industry": "automotive",
                "success": True,
            },
            {
                "label": "Circuit Diagram",
                "file": "Hydraulic_Circuit_Diagram.pdf",
                "type": "circuit_diagram",
                "route": "diagram_heavy",
                "confidence": 0.90,
                "industry": "automotive",
                "success": True,
            },
            {
                "label": "PowerPoint Presentation",
                "file": "Project_Presentation.pptx",
                "type": "presentation",
                "route": "presentation_route",
                "confidence": 0.90,
                "industry": "automotive",
                "success": True,
            },
            {
                "label": "Contract",
                "file": "Legal_Contract_2025.pdf",
                "type": "contract",
                "route": "text_default",
                "confidence": 0.90,
                "industry": "legal",
                "success": True,
            },
            {
                "label": "Financial Statement",
                "file": "Balance_Sheet_2024.xlsx",
                "type": "financial_statement",
                "route": "table_heavy",
                "confidence": 0.90,
                "industry": "finance",
                "success": True,
            },
            {
                "label": "Purchase Order",
                "file": "Purchase_Order_Engineering.xlsx",
                "type": "purchase_order",
                "route": "table_heavy",
                "confidence": 0.90,
                "industry": "automotive",
                "success": True,
            },
            {
                "label": "Policy",
                "file": "HR_Policy_Guidelines.pdf",
                "type": "policy",
                "route": "text_default",
                "confidence": 0.90,
                "industry": "automotive",
                "success": True,
            },
            {
                "label": "Manual",
                "file": "Equipment_Manual.pdf",
                "type": "manual",
                "route": "text_default",
                "confidence": 0.90,
                "industry": "automotive",
                "success": True,
            },
            {
                "label": "Research Paper",
                "file": "Clinical_Research_Paper.pdf",
                "type": "research_paper",
                "route": "text_default",
                "confidence": 0.90,
                "industry": "pharma",
                "success": True,
            },
            {
                "label": "Schematic",
                "file": "Wire_Plane_Schematic.pdf",
                "type": "schematic",
                "route": "diagram_heavy",
                "confidence": 0.90,
                "industry": "engineering",
                "success": True,
            },
        ]
    }
