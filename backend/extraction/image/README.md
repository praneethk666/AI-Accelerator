# Standalone Image Extraction Module

The Image Extraction module processes single-page bitmap graphic files (e.g. `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`, `.gif`) and integrates them into the document pipeline.

## Dependencies

* **Pillow (`PIL`)**: Open and verify image formats and save standardized previews.
* **backend.extraction.ppt.tool (`_save_image_blob`, `_detect_image_ext`)**: Reuses binary image sniffers and storage drivers to save original files and generate previews.

## Step-by-Step Logic

The pipeline entrypoint is `ImageExtractorTool::run()`:

1. **File Reading & Verification**:
   * Reads raw file bytes.
   * Invokes `Image.open().verify()` using Pillow to validate that the file is a readable image. If the check fails, the tool appends an error to `state["errors"]` and returns.
2. **Binary Image Storage**:
   * Resolves target upload directory paths `uploads/images/{doc_id}/`.
   * Sniffs magic bytes to determine the correct format extension (to prevent file corruption by saving different extensions as `.png`).
   * Saves the original image bytes as raw storage. If Pillow can decode the image, it also saves a web-compatible PNG preview file for UI rendering.
3. **Pending Vision Block Construction**:
   * Creates a `NormalizedBlock` with `type: "image_caption"` and `text: ""`.
   * Staps image paths in the block metadata:
     * `raw_image_path`: Location of original bytes.
     * `image_path`: Location of the browser-compatible PNG preview.
     * `pending_vision`: Set to `True` to tell the downstream `VisionEnrichmentTool` to describe the image.
4. **Registration**:
   * Returns the constructed block to `state["blocks"]`.
