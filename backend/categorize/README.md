# Document Categorization Tool

## Overview

The Categorization Tool is the entry point for document routing in the Document Intelligence + RAG Accelerator pipeline.

Its responsibility is to:

1. Identify the document type.
2. Determine the industry.
3. Map the document type to a processing route.
4. Write categorization metadata into the shared pipeline state.

The categorizer is designed to be vision-first, with filename and text-based signals used to improve classification accuracy and determine industry.

---

# Pipeline Contract

The pipeline calls the categorizer using:

```python
tool.run(state, config)
```

### Interface Changes (Current Version)

The `run()` function signature is:

```python
def run(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
```

- **state**: Shared pipeline state containing file_path and receiving categorization metadata
- **config**: Global configuration loaded from `config/global.yaml`
- **self**: Tool instance (passed by pipeline framework)

### Reading File Path

The categorizer reads the file path from the state:

```python
state["file_path"]
```

The file path is the ONLY input read from state.

All configuration comes from the `config` parameter (loaded from `config/global.yaml`).

---

# Global Configuration (config/global.yaml)

The categorizer uses centralized configuration from `config/global.yaml`.

**Configuration sections used:**

```yaml
categorization:
  type_to_route:          # Maps document types to processing routes
    document_type: route
  
  industry_keywords:      # Keywords for industry detection
    industry:
      - keyword1
      - keyword2
  
  confidence_thresholds:  # Confidence decision points
    categorization_low_confidence: 0.5

deployment:              # Deployment-level defaults
  default_industry: "automotive"
  client: "company_name"
```

**Key Design Principle:**

All configuration is external to the code.

No hardcoded routing rules or keyword lists exist.

The categorizer is configuration-driven.

---

The categorizer always writes the following fields:

```python
state["route"]              # Route name (e.g., "diagram_heavy")
state["document_type"]      # Type (e.g., "cad_drawing")
state["industry"]           # Industry (e.g., "automotive")
state["confidence"]         # Float 0.0-1.0 (NOT "categorization_confidence")
state["reasoning"]          # Explanation of classification decision
state["errors"]             # List of errors/warnings (may be empty)
```

**Important:** The field is `state["confidence"]`, not `state["categorization_confidence"]`.

---

# State Example

Input:

```python
state = {
    "file_path": "documents/sample.pdf",
    "errors": []
}
```

Output:

```python
{
    "route": "diagram_heavy",
    "document_type": "cad_drawing",
    "industry": "automotive",
    "confidence": 0.91,
    "reasoning": "Vision identified engineering drawing."
}
```

---

# Routing Strategy

The model predicts a document type.

The route is determined using configuration mapping.

Example:

```yaml
type_to_route:
  circuit_diagram: diagram_heavy
  cad_drawing: diagram_heavy
  schematic: diagram_heavy

  invoice: table_heavy
  financial_statement: table_heavy
  purchase_order: table_heavy

  contract: text_default
  policy: text_default
  research_paper: text_default
  report: text_default
  manual: text_default

  presentation: presentation_route
```

The model never directly decides the route.

Routes are determined only through configuration.

---

# Supported Routes

The categorizer currently supports:

```text
diagram_heavy
table_heavy
text_default
presentation_route
```

No OCR-specific route exists.

OCR decisions are handled later in the pipeline by page-level processing.

---

# Document Type Detection

Document type detection follows this order:

### 1. Filename Analysis

Example:

```text
Toyota_Circuit_Rev3.pdf
Invoice_2025.xlsx
CAD_Drawing_A12.pdf
```

Strong filename matches are used immediately.

---

### 2. Vision Classification

If filename matching is insufficient:

* Render PDF pages
* Combine pages into a stitched image
* Send image to vision model
* Predict document type

The model returns:

```python
{
    "document_type": "...",
    "confidence": 0.91,
    "reasoning": "..."
}
```

---

### 3. CAD Detection

Engineering and CAD documents receive additional detection logic through:

* Filename analysis
* Metadata extraction
* Drawing number detection
* Engineering pattern matching

---

### 4. Excel Detection

Excel documents are analyzed using workbook structure and content patterns.

Supported examples:

* Invoice
* Purchase Order
* Financial Statement

---

# Industry Detection

Industry detection is separate from document type detection.

The categorizer uses three signals:

### Signal 1: Filename

Example:

```text
Pfizer_Clinical_Trial.pdf
Toyota_Wiring_Diagram.pdf
Goldman_Q4_Report.pdf
```

---

### Signal 2: Extracted Text

Text from the first pages is scanned for industry keywords.

Examples:

```yaml
automotive:
  - toyota
  - vehicle
  - harness
  - chassis
  - ecu

pharma:
  - clinical trial
  - dosage
  - adverse event

finance:
  - revenue
  - balance sheet
  - ebitda

legal:
  - agreement
  - indemnify
  - jurisdiction
```

---

### Signal 3: Deployment Default

If no strong signal is found:

```yaml
deployment:
  default_industry: automotive
```

is used.

---

# Confidence Handling

Low-confidence classifications never stop the pipeline.

Example:

```python
if confidence < threshold:
    route = "text_default"
```

The tool adds a review message:

```python
state["errors"].append(
    "categorize: low confidence"
)
```

---

# Error Handling

The categorizer must never crash the pipeline.

All execution is wrapped in exception handling.

On failure:

```python
state["route"] = "text_default"
state["document_type"] = "report"
state["industry"] = "automotive"
state["confidence"] = 0.0
```

The exception message is added to:

```python
state["errors"]
```

---

# Configuration

The categorizer does not maintain its own configuration file.

All configuration is supplied by the pipeline through:

```text
config/global.yaml
```

The pipeline passes configuration into:

```python
run(state, config)
```

Configuration includes:

* document_types
* industries
* type_to_route
* industry_keywords
* confidence_thresholds
* deployment defaults

No local configuration is loaded inside the categorizer.

---

# Supported File Types

The categorizer accepts:

```text
PDF
Excel (.xlsx, .xls)
PowerPoint (.ppt, .pptx)
Images
```

---

# Testing Requirements

Before merging:

* Test with at least 15 varied documents.
* Include scanned PDFs.
* Include contracts.
* Include invoices.
* Include engineering drawings.
* Include CAD documents.
* Include pharma reports.
* Include presentations.
* Include Excel documents.

Required output fields must always be present:

```python
route
document_type
industry
confidence
reasoning
```

even when classification fails.

---

# Ownership

This tool is responsible only for:

* Document Type
* Industry
* Route Selection

Downstream processing, chunking, storage, retrieval, and page profiling are handled by other pipeline components.
