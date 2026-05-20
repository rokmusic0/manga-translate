from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from loguru import logger
from PIL import Image

from benchmarking import (
    JsonDict,
    load_openmantra_pages,
    ocr_result_to_dict,
    openmantra_page_slug,
    save_json,
    save_translations_markdown,
    suppress_output,
    timestamp_slug,
    translation_diagnostics_to_dict,
    translation_result_to_dict,
)
from edit_image import draw_ocr_bboxes, draw_translations, remove_ocr_regions
from logging_config import add_file_log_handlers, configure_logging
from ocr import OCR_CONFIDENCE_THRESHOLD, OCR_VL_SERVER_URL, extract_text_regions, run_ocr
from translate import LLAMA_BASE_URL, translate_images


def build_output_paths(run_dir: Path, book_title: str, page_slug: str) -> dict[str, Path]:
    page_output_dir = run_dir / "pages" / book_title / page_slug
    return {
        "page_output_dir": page_output_dir,
        "ocr_json_path": page_output_dir / f"{page_slug}_ocr.json",
        "ocr_markdown_path": page_output_dir / f"{page_slug}_ocr.md",
        "translation_markdown_path": page_output_dir / f"{page_slug}_translation.md",
        "bbox_path": page_output_dir / f"{page_slug}_ocr.png",
        "text_removed_path": page_output_dir / f"{page_slug}_text_removed.png",
        "translated_path": page_output_dir / f"{page_slug}_translated.png",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the manga pipeline on the OpenMantra dataset and save benchmark artifacts."
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/open-mantra",
        help="Directory containing the local OpenMantra dataset clone",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_runs/open-mantra",
        help="Directory where benchmark runs will be created",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run directory name",
    )
    parser.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="Optional page limit for smaller benchmark runs",
    )
    parser.add_argument(
        "--book",
        action="append",
        default=None,
        help="Restrict the run to one or more OpenMantra book titles",
    )
    parser.add_argument(
        "--show-third-party-output",
        action="store_true",
        help="Allow PaddleOCR or other libraries to print their own progress output",
    )
    parser.add_argument(
        "--skip-service-check",
        action="store_true",
        help="Skip the localhost service preflight checks",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def ensure_service_reachable(name: str, url: str) -> None:
    try:
        with urlopen(url, timeout=3):
            return
    except HTTPError:
        return
    except URLError as exc:
        raise RuntimeError(
            f"{name} is not reachable at {url}. Start the required local service first."
        ) from exc


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    dataset_dir = Path(args.dataset_dir)
    output_root = Path(args.output_dir)
    run_name = args.run_name or timestamp_slug()
    stage_timings: JsonDict = {}
    total_start = time.perf_counter()

    if not args.skip_service_check:
        logger.info("Checking local OCR and translation services")
        ensure_service_reachable("PaddleOCR-VL server", OCR_VL_SERVER_URL)
        ensure_service_reachable("llama.cpp server", f"{LLAMA_BASE_URL}/models")

    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    full_log_path, problem_log_path = add_file_log_handlers(run_dir, args.verbose)
    logger.info("Full log file: {}", full_log_path)
    logger.info("Problem image log file: {}", problem_log_path)

    logger.info("Loading OpenMantra pages from {}", dataset_dir)
    dataset_load_start = time.perf_counter()
    pages = load_openmantra_pages(
        dataset_dir,
        limit_pages=args.limit_pages,
        book_titles=set(args.book) if args.book else None,
    )
    stage_timings["dataset_load_seconds"] = time.perf_counter() - dataset_load_start

    if not pages:
        raise ValueError("No OpenMantra pages matched the requested filters")

    logger.info("Loaded {} page(s)", len(pages))
    image_paths = [page["image_path"] for page in pages]
    image_path_strings = [str(path) for path in image_paths]

    config = {
        "dataset_dir": str(dataset_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        "run_name": run_name,
        "page_count": len(pages),
        "limit_pages": args.limit_pages,
        "books": args.book or [],
        "ocr_confidence_threshold": OCR_CONFIDENCE_THRESHOLD,
        "ocr_server_url": OCR_VL_SERVER_URL,
        "llama_base_url": LLAMA_BASE_URL,
        "third_party_output_suppressed": not args.show_third_party_output,
    }
    save_json(run_dir / "config.json", config)

    logger.info("Running OCR on {} page(s)", len(image_paths))
    ocr_start = time.perf_counter()
    with suppress_output(not args.show_third_party_output):
        ocr_output, ocr_results = run_ocr(image_path_strings)
    stage_timings["ocr_seconds"] = time.perf_counter() - ocr_start

    logger.info("Running translation on {} page(s)", len(image_paths))
    translation_start = time.perf_counter()
    translations, translation_diagnostics = translate_images(image_path_strings, ocr_results)
    stage_timings["translation_seconds"] = time.perf_counter() - translation_start

    logger.info("Writing benchmark artifacts to {}", run_dir)
    artifact_start = time.perf_counter()
    pages_payload: list[JsonDict] = []
    translation_issues_payload: list[JsonDict] = []

    for page_meta, image_ocr_output, image_ocr_results, image_translations, image_translation_diagnostics in zip(
        pages,
        ocr_output,
        ocr_results,
        translations,
        translation_diagnostics,
        strict=True,
    ):
        page_slug = openmantra_page_slug(
            page_meta["book_title"],
            page_meta["page_index"],
            page_meta["image_path"],
        )
        output_paths = build_output_paths(run_dir, page_meta["book_title"], page_slug)
        page_output_dir = output_paths["page_output_dir"]
        page_output_dir.mkdir(parents=True, exist_ok=True)

        serialize_start = time.perf_counter()
        image_ocr_output.save_to_json(output_paths["ocr_json_path"].as_posix())
        image_ocr_output.save_to_markdown(output_paths["ocr_markdown_path"].as_posix())
        save_translations_markdown(
            output_paths["translation_markdown_path"],
            image_ocr_results,
            image_translations,
        )
        serialize_seconds = time.perf_counter() - serialize_start

        render_start = time.perf_counter()
        with Image.open(page_meta["image_path"]) as image:
            source_image = image.copy()

        all_text_regions = extract_text_regions(image_ocr_output, min_confidence=None)
        bbox_image = draw_ocr_bboxes(
            source_image,
            all_text_regions,
            confidence_threshold=OCR_CONFIDENCE_THRESHOLD,
        )
        cleaned_image = remove_ocr_regions(source_image, image_ocr_results)
        translated_image = draw_translations(cleaned_image, image_translations)

        bbox_image.save(output_paths["bbox_path"])
        cleaned_image.save(output_paths["text_removed_path"])
        translated_image.save(output_paths["translated_path"])
        render_seconds = time.perf_counter() - render_start

        predictions = [
            {
                **ocr_result_to_dict(ocr_result),
                "translation": (
                    image_translations[index].translation
                    if index < len(image_translations)
                    else ""
                ),
            }
            for index, ocr_result in enumerate(image_ocr_results)
        ]

        page_payload = {
            "book_title": page_meta["book_title"],
            "page_index": page_meta["page_index"],
            "image_path": str(page_meta["image_path"]),
            "relative_image_path": page_meta["relative_image_path"],
            "page_slug": page_slug,
            "ground_truth_count": len(page_meta["ground_truth"]),
            "prediction_count": len(predictions),
            "ground_truth": page_meta["ground_truth"],
            "predictions": predictions,
            "translations": [
                translation_result_to_dict(result) for result in image_translations
            ],
            "translation_diagnostics": translation_diagnostics_to_dict(
                image_translation_diagnostics
            ),
            "artifacts": {
                key: str(path.relative_to(run_dir))
                for key, path in output_paths.items()
                if key != "page_output_dir"
            },
            "timings": {
                "artifact_text_seconds": serialize_seconds,
                "render_seconds": render_seconds,
            },
        }
        pages_payload.append(page_payload)

        if image_translation_diagnostics.status != "ok":
            translation_issues_payload.append(
                {
                    "book_title": page_meta["book_title"],
                    "page_index": page_meta["page_index"],
                    "page_slug": page_slug,
                    "image_path": str(page_meta["image_path"]),
                    "relative_image_path": page_meta["relative_image_path"],
                    "translation_diagnostics": translation_diagnostics_to_dict(
                        image_translation_diagnostics
                    ),
                    "artifacts": {
                        "translation_markdown_path": str(
                            output_paths["translation_markdown_path"].relative_to(run_dir)
                        ),
                        "ocr_markdown_path": str(
                            output_paths["ocr_markdown_path"].relative_to(run_dir)
                        ),
                    },
                }
            )

    stage_timings["artifact_seconds"] = time.perf_counter() - artifact_start
    stage_timings["total_seconds"] = time.perf_counter() - total_start
    stage_timings["pages_per_second_total"] = (
        len(pages) / stage_timings["total_seconds"]
        if stage_timings["total_seconds"] > 0
        else 0.0
    )

    benchmark_results = {
        "config": config,
        "stage_timings": stage_timings,
        "pages": pages_payload,
    }
    save_json(run_dir / "benchmark_results.json", benchmark_results)
    save_json(run_dir / "stage_timings.json", stage_timings)
    save_json(run_dir / "translation_issues.json", translation_issues_payload)

    if translation_issues_payload:
        logger.warning(
            "Translation produced non-ok diagnostics for {} page(s)",
            len(translation_issues_payload),
        )
    else:
        logger.info("Translation produced ok diagnostics for all pages")

    logger.info("Saved benchmark results to {}", run_dir / "benchmark_results.json")
    logger.info("Saved translation issues to {}", run_dir / "translation_issues.json")
    logger.info("Saved stage timings to {}", run_dir / "stage_timings.json")
    logger.info(
        "Completed OpenMantra run {} in {:.2f}s",
        run_name,
        stage_timings["total_seconds"],
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
