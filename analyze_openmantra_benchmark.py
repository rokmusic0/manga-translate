from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarking import chrf_score, corpus_bleu, greedy_match_boxes, levenshtein_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a saved OpenMantra benchmark run."
    )
    parser.add_argument(
        "run_dir",
        help="Benchmark run directory created by benchmark_openmantra.py",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to match predicted and ground-truth boxes",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    results_path = run_dir / "benchmark_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    stage_timings = results["stage_timings"]
    pages = results["pages"]

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    ocr_edit_distance = 0
    ocr_reference_characters = 0
    ocr_exact_matches = 0
    matched_region_count = 0
    translated_region_count = 0

    translation_predictions: list[str] = []
    translation_references: list[str] = []

    per_page_detection: list[dict[str, float | int | str]] = []

    for page in pages:
        predicted_boxes = [tuple(pred["bbox"]) for pred in page["predictions"]]
        ground_truth_boxes = [tuple(gt["bbox"]) for gt in page["ground_truth"]]
        matches = greedy_match_boxes(
            predicted_boxes,
            ground_truth_boxes,
            iou_threshold=args.iou_threshold,
        )

        page_tp = len(matches)
        page_fp = len(predicted_boxes) - page_tp
        page_fn = len(ground_truth_boxes) - page_tp

        true_positives += page_tp
        false_positives += page_fp
        false_negatives += page_fn

        per_page_detection.append(
            {
                "page_slug": page["page_slug"],
                "book_title": page["book_title"],
                "page_index": page["page_index"],
                "true_positives": page_tp,
                "false_positives": page_fp,
                "false_negatives": page_fn,
            }
        )

        for pred_index, gt_index, _ in matches:
            prediction = page["predictions"][pred_index]
            ground_truth = page["ground_truth"][gt_index]

            predicted_ocr = prediction["ocr_text"].strip()
            reference_ocr = ground_truth["text_ja"].strip()
            ocr_edit_distance += levenshtein_distance(predicted_ocr, reference_ocr)
            ocr_reference_characters += len(reference_ocr)
            ocr_exact_matches += int(predicted_ocr == reference_ocr)
            matched_region_count += 1

            predicted_translation = prediction["translation"].strip()
            reference_translation = ground_truth["text_en"].strip()
            if predicted_translation:
                translated_region_count += 1
            translation_predictions.append(predicted_translation)
            translation_references.append(reference_translation)

    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    ocr_cer = safe_divide(ocr_edit_distance, ocr_reference_characters)
    ocr_exact_match_rate = safe_divide(ocr_exact_matches, matched_region_count)
    translation_coverage = safe_divide(
        translated_region_count,
        sum(page["ground_truth_count"] for page in pages),
    )
    bleu = corpus_bleu(translation_predictions, translation_references)
    chrf = chrf_score(translation_predictions, translation_references)

    summary = {
        "page_count": len(pages),
        "ground_truth_region_count": sum(page["ground_truth_count"] for page in pages),
        "predicted_region_count": sum(page["prediction_count"] for page in pages),
        "matched_region_count": matched_region_count,
        "detection": {
            "iou_threshold": args.iou_threshold,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "ocr": {
            "character_error_rate": ocr_cer,
            "exact_match_rate": ocr_exact_match_rate,
            "reference_character_count": ocr_reference_characters,
        },
        "translation": {
            "bleu": bleu,
            "chrf": chrf,
            "coverage": translation_coverage,
            "evaluated_region_count": len(translation_predictions),
        },
        "timings": {
            **stage_timings,
            "average_seconds_per_page_total": safe_divide(
                stage_timings["total_seconds"], len(pages)
            ),
            "average_seconds_per_page_ocr": safe_divide(
                stage_timings["ocr_seconds"], len(pages)
            ),
            "average_seconds_per_page_translation": safe_divide(
                stage_timings["translation_seconds"], len(pages)
            ),
            "average_seconds_per_page_artifacts": safe_divide(
                stage_timings["artifact_seconds"], len(pages)
            ),
        },
        "per_page_detection": per_page_detection,
    }

    analysis_json_path = run_dir / "analysis.json"
    analysis_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    analysis_markdown = "\n".join(
        [
            "# OpenMantra benchmark analysis",
            "",
            "## Counts",
            "",
            f"- Pages: `{summary['page_count']}`",
            f"- Ground-truth text regions: `{summary['ground_truth_region_count']}`",
            f"- Predicted text regions: `{summary['predicted_region_count']}`",
            f"- Matched text regions: `{summary['matched_region_count']}`",
            "",
            "## Detection",
            "",
            f"- IoU threshold: `{summary['detection']['iou_threshold']}`",
            f"- Precision: `{summary['detection']['precision']:.4f}`",
            f"- Recall: `{summary['detection']['recall']:.4f}`",
            f"- F1: `{summary['detection']['f1']:.4f}`",
            "",
            "## OCR",
            "",
            f"- Character error rate: `{summary['ocr']['character_error_rate']:.4f}`",
            f"- Exact match rate: `{summary['ocr']['exact_match_rate']:.4f}`",
            "",
            "## Translation",
            "",
            f"- BLEU: `{summary['translation']['bleu']:.2f}`",
            f"- chrF: `{summary['translation']['chrf']:.2f}`",
            f"- Coverage: `{summary['translation']['coverage']:.4f}`",
            "",
            "## Timings",
            "",
            f"- Dataset load seconds: `{summary['timings']['dataset_load_seconds']:.2f}`",
            f"- OCR seconds: `{summary['timings']['ocr_seconds']:.2f}`",
            f"- Translation seconds: `{summary['timings']['translation_seconds']:.2f}`",
            f"- Artifact seconds: `{summary['timings']['artifact_seconds']:.2f}`",
            f"- Total seconds: `{summary['timings']['total_seconds']:.2f}`",
            f"- Average total seconds/page: `{summary['timings']['average_seconds_per_page_total']:.2f}`",
            f"- Average OCR seconds/page: `{summary['timings']['average_seconds_per_page_ocr']:.2f}`",
            f"- Average translation seconds/page: `{summary['timings']['average_seconds_per_page_translation']:.2f}`",
            f"- Average artifact seconds/page: `{summary['timings']['average_seconds_per_page_artifacts']:.2f}`",
            "",
        ]
    )
    (run_dir / "analysis.md").write_text(analysis_markdown, encoding="utf-8")

    print(f"Saved analysis to {analysis_json_path}")
    print(f"Saved analysis to {run_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
