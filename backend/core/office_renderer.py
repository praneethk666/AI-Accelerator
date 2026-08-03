"""Direct office document rendering utility.

Renders PPTX slides directly to JPEGs (p1.jpg, p2.jpg, etc.) in the background
during ingestion, supporting both Windows COM objects (natively) and
headless LibreOffice (as fallback).
"""
import os
import logging
import subprocess

logger = logging.getLogger(__name__)


def _render_pptx_slides_com(input_path: str, output_dir: str) -> int:
    import win32com.client
    powerpoint = win32com.client.DispatchEx("PowerPoint.Application")
    presentations = None
    deck = None
    slides = None
    slide = None
    try:
        presentations = powerpoint.Presentations
        # Try opening hidden first (WithWindow=False)
        try:
            deck = presentations.Open(os.path.abspath(input_path), ReadOnly=True, WithWindow=False)
        except Exception as e:
            logger.warning("PowerPoint open hidden failed: %s. Retrying with default window context...", e)
            deck = presentations.Open(os.path.abspath(input_path), ReadOnly=True)

        slides = deck.Slides
        total_slides = slides.Count
        
        # Try fast native batch export first!
        try:
            deck.Export(os.path.abspath(output_dir), "JPG")
            # PowerPoint native export saves slides inside output_dir as Slide1.JPG, Slide2.JPG, etc.
            # Let's check and rename them to p1.jpg, p2.jpg, etc.
            import glob
            slide_files = glob.glob(os.path.join(output_dir, "Slide*.JPG")) + glob.glob(os.path.join(output_dir, "Slide*.jpg"))
            # If we found native export files, rename them
            if len(slide_files) >= total_slides:
                for s_path in slide_files:
                    s_name = os.path.basename(s_path)
                    # extract digits
                    digits = "".join([c for c in s_name if c.isdigit()])
                    if digits:
                        idx = int(digits)
                        dest_path = os.path.join(output_dir, f"p{idx}.jpg")
                        # Remove existing if any
                        if os.path.isfile(dest_path):
                            os.remove(dest_path)
                        os.rename(s_path, dest_path)
                logger.info("Successfully rendered %d slides to JPEG using PowerPoint COM native Export", total_slides)
                return total_slides
        except Exception as export_err:
            logger.warning("PowerPoint native batch Export failed: %s. Falling back to individual slide export loop...", export_err)

        # Export each slide to the output directory via loop
        for i in range(1, total_slides + 1):
            slide = slides(i)
            out_path = os.path.join(output_dir, f"p{i}.jpg")
            if os.path.isfile(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            slide.Export(os.path.abspath(out_path), "JPG")
            slide = None
        
        logger.info("Successfully rendered %d slides to JPEG using PowerPoint COM slide loop", total_slides)
        return total_slides
    finally:
        # Release all COM wrapper references. Do not call deck.Close() or powerpoint.Quit()
        # explicitly as they can trigger early termination of the PowerPoint process,
        # which causes other active COM references (like slides/presentations) to become
        # disconnected and raise RPC_E_DISCONNECTED exceptions during finalization.
        slide = None
        slides = None
        deck = None
        presentations = None
        powerpoint = None
        import gc
        gc.collect()


def render_pptx_slides(input_path: str, output_dir: str) -> int:
    """Render slides of a PPTX presentation to JPEG/PNG images.

    Images are saved as p1.jpg, p2.jpg, etc. under output_dir.
    Returns the total count of rendered slides.
    """
    os.makedirs(output_dir, exist_ok=True)
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in (".pptx", ".ppt"):
        return 0

    # Try Windows COM object first (since user is on Windows)
    if os.name == "nt":
        try:
            import pythoncom
            pythoncom.CoInitialize()
            try:
                logger.info("Attempting PowerPoint COM slide rendering for %s", os.path.basename(input_path))
                return _render_pptx_slides_com(input_path, output_dir)
            finally:
                import gc
                gc.collect()
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Windows COM slide rendering failed: %s. Trying LibreOffice...", e)

    # Try LibreOffice/soffice fallback (converts presentation slides directly to images)
    try:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        cmd = ["soffice", "--headless", "--convert-to", "png", "--outdir", output_dir, input_path]
        logger.info("Running LibreOffice command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Scan for generated image files
        import glob
        pattern = os.path.join(output_dir, f"{base_name}*.png")
        files = sorted(glob.glob(pattern))
        if files:
            for idx, file in enumerate(files, start=1):
                from PIL import Image
                img = Image.open(file)
                out_path = os.path.join(output_dir, f"p{idx}.jpg")
                img.convert("RGB").save(out_path, "JPEG", quality=85)
                # clean up the temporary png
                try:
                    os.remove(file)
                except Exception:
                    pass
            logger.info("Successfully rendered %d slides to JPEG using LibreOffice", len(files))
            return len(files)
    except Exception as e:
        logger.warning("LibreOffice slide rendering failed: %s", e)

    return 0
