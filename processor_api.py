"""
processor_api.py

Unified processing module for Dolphin OCR‑VQA model:
- process_element (single text/table/formula)
- process_single_image + process_elements (page‑level + element‑level)
- process_document (PDF or image)
- generate_markdown
"""
import os
import glob
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from chat import DOLPHIN
from all_utils.utils import (
    convert_pdf_to_images,
    prepare_image,
    parse_layout_string,
    process_coordinates,
    save_outputs,
    save_figure_to_local,
    crop_margin,
    save_combined_pdf_results,
    ImageDimensions,
)


def process_element(
    image_path: str,
    model: DOLPHIN,
    element_type: str,
    save_dir: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a single element image (text, table, formula).
    """
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = crop_margin(pil_image)

    if element_type == "table":
        prompt, label = "Parse the table in the image.", "tab"
    elif element_type == "formula":
        prompt, label = "Read text in the image.", "formula"
    else:
        prompt, label = "Read text in the image.", "text"

    raw = model.chat(prompt, pil_image)
    text = raw.strip()
    recognition = [{"label": label, "text": text}]

    if save_dir:
        save_outputs(recognition, image_path, save_dir)
    return text, recognition


def process_single_image(
    image: Image.Image,
    model: DOLPHIN,
    save_dir: str,
    image_name: str,
    max_batch_size: int,
    save_individual: bool = True,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Process a single image (from file or PDF page).
    """
    # Stage 1: layout parsing
    layout_output = model.chat(
        "Parse the reading order of this document.", image
    )

    # Stage 2: element parsing
    padded_image, dims = prepare_image(image)
    recognition_results = process_elements(
        layout_output, padded_image, dims, model, max_batch_size, save_dir, image_name
    )

    json_path: Optional[str] = None
    if save_individual:
        dummy_img = f"{image_name}.jpg"
        json_path = save_outputs(recognition_results, dummy_img, save_dir)
    return json_path, recognition_results


def process_elements(
    layout_results: str,
    padded_image: np.ndarray,
    dims: ImageDimensions,
    model: DOLPHIN,
    max_batch_size: int,
    save_dir: Optional[str] = None,
    image_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parse all document elements with parallel decoding
    """
    elements = parse_layout_string(layout_results)
    text_table_items: List[Dict[str, Any]] = []
    figure_items: List[Dict[str, Any]] = []
    prev_box = None
    order = 0

    for coords, label in elements:
        try:
            x1, y1, x2, y2, ox1, oy1, ox2, oy2, prev_box = process_coordinates(
                coords, padded_image, dims, prev_box
            )
            crop = padded_image[y1:y2, x1:x2]
            if crop.size == 0:
                order += 1
                continue
            if label == "fig":
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                fn = save_figure_to_local(pil_crop, save_dir, image_name or "", order)
                figure_items.append({
                    "label": label,
                    "text": f"![Figure](figures/{fn})",
                    "figure_path": f"figures/{fn}",
                    "bbox": [ox1, oy1, ox2, oy2],
                    "reading_order": order,
                })
            else:
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                prompt = "Parse the table in the image." if label == "tab" else "Read text in the image."
                text_table_items.append({
                    "crop": pil_crop,
                    "prompt": prompt,
                    "label": label,
                    "bbox": [ox1, oy1, ox2, oy2],
                    "reading_order": order,
                })
            order += 1
        except Exception:
            order += 1
            continue

    results: List[Dict[str, Any]] = figure_items
    if text_table_items:
        crops = [item["crop"] for item in text_table_items]
        prompts = [item["prompt"] for item in text_table_items]
        batch = model.chat(prompts, crops, max_batch_size=max_batch_size)
        for i, res in enumerate(batch):
            itm = text_table_items[i]
            results.append({
                "label": itm["label"],
                "bbox": itm["bbox"],
                "text": res.strip(),
                "reading_order": itm["reading_order"],
            })
    results.sort(key=lambda x: x.get("reading_order", 0))
    return results


def process_document(
    document_path: str,
    model: DOLPHIN,
    save_dir: str,
    max_batch_size: int,
) -> Tuple[Any, Any]:
    """
    Parse a document (PDF or image) into recognition results.
    """
    ext = os.path.splitext(document_path)[1].lower()
    if ext == ".pdf":
        img_list = convert_pdf_to_images(document_path)
        pages: List[Dict[str, Any]] = []
        base = os.path.splitext(os.path.basename(document_path))[0]
        for idx, img in enumerate(img_list, 1):
            name = f"{base}_page_{idx:03d}"
            _, elems = process_single_image(
                img, model, save_dir, name, max_batch_size, save_individual=False
            )
            pages.append({"page": idx, "elements": elems})
        json_path = save_combined_pdf_results(pages, document_path, save_dir)
        return json_path, pages
    else:
        img = Image.open(document_path).convert("RGB")
        base = os.path.splitext(os.path.basename(document_path))[0]
        return process_single_image(img, model, save_dir, base, max_batch_size)


def generate_markdown(results: Any) -> str:
    """
    Convert recognition results into Markdown string.
    """
    md: List[str] = []
    if isinstance(results, list) and results and "page" in results[0]:
        for pg in results:
            md.append(f"## Page {pg['page']}\n")
            for el in pg['elements']:
                if el['label'] == 'fig':
                    md.append(f"![Figure]({el['figure_path']})\n")
                else:
                    md.append(f"- **{el['label']}**: {el['text']}\n")
    else:
        for el in results:
            if el['label'] == 'fig':
                md.append(f"![Figure]({el['figure_path']})\n")
            else:
                md.append(f"- **{el['label']}**: {el['text']}\n")
    return "\n".join(md)
