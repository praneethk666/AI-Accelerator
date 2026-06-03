# backend/vision/vision_enrichment.py

from .pdf_cropper import PDFCropper
from .vision_client import VisionClient
from .block_builder import build_image_caption_block
from .timeout import (
    run_with_timeout,
    TimeoutException,
)


class VisionEnrichmentTool:

    name = "vision_enrichment"

    def __init__(self, model_name=None):

        self.cropper = PDFCropper()

        self.vision_client = VisionClient(
            model_name=model_name
        )

    def run(self, state, config):

        page_profiles = state.get(
            "page_profiles",
            [],
        )

        blocks = state.setdefault(
            "blocks",
            [],
        )

        errors = state.setdefault(
            "errors",
            [],
        )

        vision_cfg = config.get(
            "vision",
            {},
        )

        timeout_s = vision_cfg.get(
            "timeout_s",
            45,
        )

        dpi = vision_cfg.get(
            "dpi",
            200,
        )

        for profile in page_profiles:

            page_number = profile["page_number"]

            for image in profile.get(
                "images",
                [],
            ):

                if not image.get(
                    "significant",
                    False,
                ):
                    continue

                bbox = image["bbox"]

                try:

                    image_bytes = (
                        self.cropper.crop_region(
                            pdf_path=state[
                                "file_path"
                            ],
                            page_number=page_number,
                            bbox=bbox,
                            dpi=dpi,
                        )
                    )

                    caption_json = (
                        run_with_timeout(
                            self.vision_client.describe,
                            timeout_s,
                            image_bytes,
                            config=vision_cfg,
                        )
                    )

                    block = (
                        build_image_caption_block(
                            state=state,
                            page_number=page_number,
                            bbox=bbox,
                            caption_json_str=caption_json,
                        )
                    )

                    blocks.append(block)

                except TimeoutException:

                    errors.append(
                        f"Vision timeout page {page_number}, bbox {bbox}"
                    )

                except Exception as e:

                    errors.append(
                        f"Vision error page {page_number}: {str(e)}"
                    )

        return state