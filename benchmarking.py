from __future__ import annotations

import json
import math
import os
import re
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

from models import OCRResult, TranslationDiagnostics, TranslationResult

type JsonDict = dict[str, Any]
type BBox = tuple[int, int, int, int]

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_translations_markdown(
    path: Path,
    ocr_results: list[OCRResult],
    translations: list[TranslationResult],
) -> None:
    content = "\n\n".join(
        f"{ocr_result.text}\n{translations[i].translation if i < len(translations) else ''}"
        for i, ocr_result in enumerate(ocr_results)
    )
    path.write_text(content, encoding="utf-8")


@contextmanager
def suppress_output(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def load_openmantra_pages(
    dataset_dir: Path,
    *,
    limit_pages: int | None = None,
    book_titles: set[str] | None = None,
) -> list[JsonDict]:
    annotation_path = dataset_dir / "annotation.json"
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))

    pages: list[JsonDict] = []

    for book in dataset:
        book_title = book["book_title"]
        if book_titles is not None and book_title not in book_titles:
            continue

        for page in book["pages"]:
            relative_image_path = page["image_paths"]["ja"]
            ground_truth = [
                {
                    "bbox": [
                        int(text_item["x"]),
                        int(text_item["y"]),
                        int(text_item["x"] + text_item["w"]),
                        int(text_item["y"] + text_item["h"]),
                    ],
                    "text_ja": text_item["text_ja"],
                    "text_en": text_item["text_en"],
                    "text_zh": text_item.get("text_zh", ""),
                }
                for text_item in page["text"]
            ]

            image_path = dataset_dir / relative_image_path
            pages.append(
                {
                    "book_title": book_title,
                    "page_index": int(page["page_index"]),
                    "image_path": image_path,
                    "relative_image_path": relative_image_path,
                    "ground_truth": ground_truth,
                }
            )

            if limit_pages is not None and len(pages) >= limit_pages:
                return pages

    return pages


def openmantra_page_slug(book_title: str, page_index: int, image_path: Path) -> str:
    return f"{book_title}_p{page_index:03d}_{image_path.stem}"


def ocr_result_to_dict(result: OCRResult) -> JsonDict:
    return {
        "bbox": list(result.bbox),
        "confidence": result.confidence,
        "ocr_text": result.text,
    }


def translation_result_to_dict(result: TranslationResult) -> JsonDict:
    return {
        "bbox": list(result.ocr_result.bbox),
        "confidence": result.ocr_result.confidence,
        "ocr_text": result.ocr_result.text,
        "translation": result.translation,
    }



def translation_diagnostics_to_dict(result: TranslationDiagnostics) -> JsonDict:
    return {
        "expected_count": result.expected_count,
        "actual_count": result.actual_count,
        "status": result.status,
        "detail": result.detail,
    }


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_left = max(ax1, bx1)
    inter_top = max(ay1, by1)
    inter_right = min(ax2, bx2)
    inter_bottom = min(ay2, by2)

    inter_width = max(0, inter_right - inter_left)
    inter_height = max(0, inter_bottom - inter_top)
    intersection = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def greedy_match_boxes(
    predicted_boxes: list[BBox],
    ground_truth_boxes: list[BBox],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    candidate_matches: list[tuple[float, int, int]] = []

    for pred_index, pred_box in enumerate(predicted_boxes):
        for gt_index, gt_box in enumerate(ground_truth_boxes):
            iou = bbox_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                candidate_matches.append((iou, pred_index, gt_index))

    candidate_matches.sort(reverse=True)

    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou, pred_index, gt_index in candidate_matches:
        if pred_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
        matches.append((pred_index, gt_index, iou))

    return matches


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))

    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current

    return previous[-1]


def tokenize_english(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def get_ngram_counts(tokens: list[str], order: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if len(tokens) < order:
        return counts

    for index in range(len(tokens) - order + 1):
        ngram = tuple(tokens[index : index + order])
        counts[ngram] = counts.get(ngram, 0) + 1

    return counts


def corpus_bleu(
    predictions: list[str],
    references: list[str],
    *,
    max_order: int = 4,
) -> float:
    if not predictions or not references:
        return 0.0

    clipped_totals = [0] * max_order
    total_counts = [0] * max_order
    prediction_length = 0
    reference_length = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = tokenize_english(prediction)
        ref_tokens = tokenize_english(reference)
        prediction_length += len(pred_tokens)
        reference_length += len(ref_tokens)

        for order in range(1, max_order + 1):
            pred_counts = get_ngram_counts(pred_tokens, order)
            ref_counts = get_ngram_counts(ref_tokens, order)
            total_counts[order - 1] += sum(pred_counts.values())

            for ngram, count in pred_counts.items():
                clipped_totals[order - 1] += min(count, ref_counts.get(ngram, 0))

    if prediction_length == 0:
        return 0.0

    precisions: list[float] = []
    for clipped_total, total_count in zip(clipped_totals, total_counts):
        precisions.append((clipped_total + 1.0) / (total_count + 1.0))

    log_precision_sum = sum(math.log(p) for p in precisions) / max_order
    brevity_penalty = 1.0
    if prediction_length < reference_length:
        brevity_penalty = math.exp(1.0 - (reference_length / prediction_length))

    return brevity_penalty * math.exp(log_precision_sum) * 100.0


def chrf_score(
    predictions: list[str],
    references: list[str],
    *,
    beta: float = 2.0,
    max_char_order: int = 6,
) -> float:
    if not predictions or not references:
        return 0.0

    beta_sq = beta * beta
    order_scores: list[float] = []

    for order in range(1, max_char_order + 1):
        overlap_total = 0
        pred_total = 0
        ref_total = 0

        for prediction, reference in zip(predictions, references):
            pred_chars = list(prediction)
            ref_chars = list(reference)
            pred_counts = get_ngram_counts(pred_chars, order)
            ref_counts = get_ngram_counts(ref_chars, order)

            pred_total += sum(pred_counts.values())
            ref_total += sum(ref_counts.values())
            overlap_total += sum(
                min(count, ref_counts.get(ngram, 0))
                for ngram, count in pred_counts.items()
            )

        if pred_total == 0 or ref_total == 0 or overlap_total == 0:
            order_scores.append(0.0)
            continue

        precision = overlap_total / pred_total
        recall = overlap_total / ref_total
        order_scores.append(
            ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)
        )

    return (sum(order_scores) / max_char_order) * 100.0
