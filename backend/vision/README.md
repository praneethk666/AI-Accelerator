# Vision Enrichment Module

The Vision module extracts visual regions from documents, crops figures, and captions them using multimodal models.

## Dependencies

* **PyMuPDF (`fitz`)**: Coordinates document page rendering and coordinate cropping.
* **backend.core.vision_client (`describe_image`)**: Coordinates API requests to Gemini, OpenAI, or local Ollama vision models.
* **Pillow (`PIL`)**: Processes cropped image pixels and saves files to disk.
* **concurrent.futures**: Coordinates multi-threaded API requests.

## Step-by-Step Logic

The process runs in `VisionEnrichmentTool::run()`:

1. **Page Profiling**:
   * Scans `state["page_profiles"]` to find structural bounding boxes matching figures, charts, and vectors.
2. **Page Cropping (`pdf_cropper.py`)**:
   * PyMuPDF opens the PDF document at `state["file_path"]`.
   * Crops bounding box areas at a configured resolution (default: 150 DPI).
   * Saves crops to `uploads/images/{doc_id}/{block_id}.png`.
3. **Prompt Enrichment (`_prompt_with_context`)**:
   * Before sending an image to the model, the tool extracts surrounding text from the same page:
     $$\text{Final Prompt} = \text{System Prompt} + \text{"\nSURROUNDING PAGE TEXT: "} + \text{page\_text}$$
   * This provides the VLM with surrounding context, helping it identify part numbers, model codes, and captions referenced in the page prose.
4. **Duplicate Deduplication**:
   * Computes the MD5 hash of cropped images. If a duplicate hash is detected, it reuses the existing caption instead of making another API call.
5. **Parallel Processing**:
   * Submits caption requests to a `ThreadPoolExecutor` (default: 4 threads).
6. **Execution Safeguards (`timeout.py`)**:
   * API requests are wrapped in timeout blocks (`run_with_timeout`) to prevent hung network calls from blocking the FastAPI server event loop.
