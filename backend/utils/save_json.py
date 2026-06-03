import os
import json
from dataclasses import asdict


def save_page_profiles(profiles, pdf_path):
    """
    Save PageProfile list as JSON.
    """

    os.makedirs("output/page_profiles", exist_ok=True)

    pdf_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    output_file = (
        f"output/page_profiles/"
        f"{pdf_name}_page_profiles.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(profile) for profile in profiles],
            f,
            indent=2
        )

    print(f"Saved: {output_file}")


def save_blocks(blocks, pdf_path):
    """
    Save NormalizedBlock list as JSON.
    """

    os.makedirs("output/blocks", exist_ok=True)

    pdf_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    output_file = (
        f"output/blocks/"
        f"{pdf_name}_blocks.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(block) for block in blocks],
            f,
            indent=2
        )

    print(f"Saved: {output_file}")