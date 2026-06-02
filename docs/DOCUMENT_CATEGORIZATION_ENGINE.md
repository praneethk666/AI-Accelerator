# My Document Categorization & Routing Engine
## Abhishek — Sprint 1

**Owner:** Me (Abhishek)  
**Branch:** `feat/abhishek-doc-categorize`  
**Status:** Ready to Build

---

## What I'm Building (Simple)

I'm the **first checkpoint** for every document. 

My job: **Look → Understand → Route**

```
Document arrives
    ↓
I read filename & first page
    ↓
I decide what type it is
    ↓
I pick which processing lane
    ↓
Pipeline runs correct tools
```


---

## What I Do ✅ & What I Don't ❌

### I DO:
- Identify document type (invoice, contract, circuit diagram, etc.)
- Detect industry (automotive, finance, legal, etc.)
- Pick the right processing route
- Score my confidence
- Handle uncertain cases gracefully

### I DON'T:
- Extract full PDFs (that's Manoj)
- Generate embeddings (that's Karthii)
- Answer questions (that's the LLM)
- Train ML models (rule-based only)

---

## Why This Matters

**Without me:**
```
Invoice → Vision AI (waste) → OCR (waste) → Diagram detection (waste)
Cost: $$$, Speed: slow, Accuracy: bad ❌
```

**With me:**
```
Invoice → [ABHISHEK] → "Route: table_heavy"
→ Table extraction ON, Vision OFF
Cost: $, Speed: fast, Accuracy: good ✅
```

---

## My 5 Routes

| Route | When I Use | Example Docs |
|-------|-----------|--------------|
| **diagram_heavy** | Circuit diagrams, CAD, schematics | Toyota_Circuit.pdf |
| **table_heavy** | Invoices, spreadsheets, financial reports | Invoice_Q4.pdf |
| **text_default** | Contracts, policies, research papers | Contract_2025.pdf |
| **ocr_heavy** | Scanned PDFs, image files, old docs | Scanned_1995.pdf |
| **presentation_route** | PowerPoint, slide decks | Presentation.pptx |

---

## My 15 Document Types

- Invoice
- Contract
- Report
- Specification
- CAD Drawing
- Circuit Diagram
- Presentation
- Spreadsheet
- Manual
- Policy
- Research Paper
- Purchase Order
- Financial Statement
- Resume
- General (Unknown)

---

## My 9 Industries

- Automotive
- Finance
- Legal
- Healthcare
- Manufacturing
- Engineering
- Insurance
- Education
- General

---

## How I Categorize (Simple)

### Step 1: Extract Signals
From: filename, first page text, page layout

### Step 2: Score Each Type
```
If filename has "invoice": +5 points
If page has "Invoice Number": +4 points
If page has "Tax": +3 points
If page has tables: +2 points
...
Total: 20 points for invoice
```

### Step 3: Pick Winner
Best type = highest score

### Step 4: Calculate Confidence
```
confidence = (best_score - second_best_score) / best_score
```

### Step 5: Handle Low Confidence
```
HIGH (0.8+)   → Use result normally
MEDIUM (0.5+) → Use result, but flag it
LOW (<0.5)    → Default to text_default + flag for review
```

---

## My Output (Every Document)

```json
{
  "document_type": "invoice",
  "industry": "finance",
  "route": "table_heavy",
  "confidence": 0.91,
  "needs_review": false,
  "reasoning": [
    "filename: 'Invoice' (+5)",
    "keyword: 'Bill To' (+4)",
    "layout: 'tables' (+2)"
  ]
}
```

This goes into `state["route"]` so Karthii knows what to do next.

---

## My Folder Structure

```
backend/categorize/
├── taxonomy.py       # My document types, industries, routes
├── classifier.py     # My scoring logic
├── categorize_tool.py # My Tool interface (connects to pipeline)
├── config.yaml       # My keywords, thresholds, settings
└── README.md         # How to use my module
```

---

## My Config (Simplified)

```yaml
categorization:
  high_confidence: 0.8
  medium_confidence: 0.5
  default_route: "text_default"
  
keywords:
  invoice: ["invoice", "bill to", "tax", "amount due"]
  circuit_diagram: ["circuit", "wiring", "voltage", "resistor"]
  contract: ["agreement", "terms", "party a", "signature"]

industries:
  automotive: ["automotive", "toyota", "vehicle", "car"]
  finance: ["finance", "financial", "accounting"]
```

---

## My Implementation (Code Sketch)

```python
class DocumentClassifier:
    def extract_signals(self, file_path):
        # Get filename, first page text, layout signals
        return signals
    
    def score_document(self, signals):
        # Score each document type
        # Return scores dict
        return scores
    
    def classify(self, file_path):
        signals = self.extract_signals(file_path)
        scores = self.score_document(signals)
        best = max(scores)
        confidence = calculate(best, scores)
        
        return {
            "type": best_type,
            "industry": detected_industry,
            "route": type_to_route[best_type],
            "confidence": confidence,
            "needs_review": confidence < 0.5
        }

class CategorizeTool(Tool):
    def run(self, state, config):
        result = self.classifier.classify(state["file_path"])
        state["document_type"] = result["type"]
        state["industry"] = result["industry"]
        state["route"] = result["route"]
        state["confidence"] = result["confidence"]
        return state
```

---

## My Tests

```python
# Test 1: Scoring works
def test_score(): 
    scores = classifier.score_document(signals)
    assert scores["invoice"] > scores["report"]

# Test 2: Confidence is valid
def test_confidence():
    conf = classifier.calculate_confidence(scores)
    assert 0 <= conf <= 1

# Test 3: Low confidence handled
def test_low_conf():
    result = classifier.classify("unclear.pdf")
    if result["confidence"] < 0.5:
        assert result["needs_review"] == True
        assert result["route"] == "text_default"

# Test 4: Speed
def test_speed():
    start = time.time()
    classifier.classify("doc.pdf")
    elapsed = time.time() - start
    assert elapsed < 1.0  # Must be fast
```

---

## What I'm Delivering

✅ **Taxonomy**: List of all document types, industries, routes

✅ **Classifier Tool**: Python module that works (categorize_tool.py)

✅ **Test Results**: Tested on 15+ sample documents with results

✅ **Config File**: All keywords, thresholds in config.yaml

✅ **README**: How to use my module + examples

✅ **Demo Script**: Show document → categorize → route works

---

## How I Connect to Others

```
My state["route"]
    ↓
Karthii reads it → enables/disables tools
    ↓
Manoj knows if it's diagrams → uses right extraction
    ↓
Vishal knows if it's diagram_heavy → calls vision API
    ↓
Vinod knows industry → applies correct filters
    ↓
LLM gets right context → better answers
```

---

## My Better Ideas (Optional)

1. **Store reasoning**: Show WHY I made each decision (for debugging)
2. **Multiple types**: Support `primary_type` + `secondary_type` (many docs fit multiple)
3. **Confidence bands**: Return "HIGH", "MEDIUM", "LOW" (easier for downstream)
4. **ML upgrade path**: Sprint 1 = rules, Sprint 2+ = hybrid with small LLM

---

## When I'm Done ✅

1. Taxonomy is clear and documented
2. My tool runs: `classifier.classify(file_path)` returns correct output
3. Tested on 15 diverse documents, confidence ≥ 0.7
4. Low confidence cases handled (default to text_default + flag)
5. Karthii can read my `state["route"]` and know what to do
6. Everything documented so others understand my code

---

## My Simple Timeline

- **Day 1**: Design taxonomy (types, industries, routes)
- **Day 2-3**: Build classifier.py (scoring logic)
- **Day 4**: Build categorize_tool.py (Tool interface)
- **Day 5**: Test on 15 documents
- **Day 6**: Write README + docs
- **Day 7**: Pull Request to main

No fixed dates. Quality over speed.

---

## My Key Files to Create

| File | What I Put There |
|------|-----------------|
| `backend/categorize/taxonomy.py` | Document types, industries, routes |
| `backend/categorize/classifier.py` | Scoring logic |
| `backend/categorize/categorize_tool.py` | Tool interface |
| `backend/categorize/config.yaml` | Keywords, thresholds |
| `backend/categorize/README.md` | How to use |
| `tests/test_categorize.py` | My tests |
| `tests/categorization_results.json` | Results on 15 docs |

---

## What I Return (Key Output)

Every document returns:
```json
{
  "document_type": "...",
  "industry": "...",
  "route": "...",
  "confidence": 0.0-1.0,
  "needs_review": true/false
}
```

This tells the pipeline exactly how to process the document.

---

**Owner:** Abhishek (Me)  
**Status:** Ready to Build  
**Let's Go!** 🚀

