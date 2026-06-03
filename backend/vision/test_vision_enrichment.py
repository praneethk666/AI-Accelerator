# backend/vision/test_vision_enrichment.py

import json
from backend.vision.vision_enrichment import VisionEnrichmentTool


def main():

    state = {
        "document_id": "doc123",

        # Change this to your PDF path
        "file_path": "test-data/.pdf",

        "page_profiles": [
            {
                "page_number": 1,
                "kind": "mixed",
                "text_len": 1280,
                "has_vector_graphics": True,
                "table_hint": False,

                "images": [
                    {
                        "bbox": [100, 100, 500, 500],
                        "width": 400,
                        "height": 400,
                        "significant": True,
                    }
                ],
            }
        ],
    }

    config = {
        "vision": {
            "timeout_s": 45,
            "dpi": 200,
        }
    }

    print("\n" + "=" * 80)
    print("RUNNING VISION ENRICHMENT")
    print("=" * 80)

    tool = VisionEnrichmentTool()

    result = tool.run(state, config)

    print("\n" + "=" * 80)
    print("GENERATED BLOCKS")
    print("=" * 80)

    print(
        json.dumps(
            result.get("blocks", []),
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        f"\nTotal Blocks Generated: "
        f"{len(result.get('blocks', []))}"
    )

    if result.get("errors"):

        print("\n" + "=" * 80)
        print("ERRORS")
        print("=" * 80)

        for err in result["errors"]:
            print(f" - {err}")

    else:

        print("\n" + "=" * 80)
        print("NO ERRORS")
        print("=" * 80)


if __name__ == "__main__":
    main()