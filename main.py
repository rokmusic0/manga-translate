import argparse
from pathlib import Path
import dotenv

from loguru import logger
from PIL import Image

from edit_image import draw_ocr_bboxes, draw_translations, remove_ocr_regions
from logging_config import configure_logging
from models import OCRResult, TranslationResult
from ocr import OCR_CONFIDENCE_THRESHOLD, extract_text_regions, run_ocr
from translate import translate_images


dotenv.load_dotenv()


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_paths(inputs: list[str]) -> list[Path]:
    logger.info("Collecting input images")
    image_paths: list[Path] = []

    for input_path in inputs:
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_dir():
            dir_images = sorted(file for file in path.iterdir() if is_image_file(file))
            logger.info("Found {} image(s) in directory {}", len(dir_images), path)
            logger.debug("Directory images for {}: {}", path, dir_images)
            image_paths.extend(dir_images)
        elif is_image_file(path):
            logger.debug("Using input image {}", path)
            image_paths.append(path)
        else:
            raise ValueError(f"Input path is not an image-like file: {path}")

    deduped_paths: list[Path] = []
    seen: set[Path] = set()
    for path in image_paths:
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        deduped_paths.append(path)

    if not deduped_paths:
        raise ValueError("No image files found in the provided inputs")

    logger.info("Collected {} unique image(s)", len(deduped_paths))
    logger.debug("Final image list: {}", deduped_paths)
    return deduped_paths


def build_output_paths(
    output_dir: Path, image_path: Path
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    image_output_dir = output_dir / image_path.stem
    ocr_json_path = image_output_dir / f"{image_path.stem}_ocr.json"
    ocr_markdown_path = image_output_dir / f"{image_path.stem}_ocr.md"
    translation_markdown_path = image_output_dir / f"{image_path.stem}_translation.md"
    bbox_path = image_output_dir / f"{image_path.stem}_ocr.png"
    text_removed_path = image_output_dir / f"{image_path.stem}_text_removed.png"
    translated_path = output_dir / f"{image_path.stem}.png"
    return (
        image_output_dir,
        ocr_json_path,
        ocr_markdown_path,
        translation_markdown_path,
        bbox_path,
        text_removed_path,
        translated_path,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR, translation, and image editing on one or more manga images."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image files and/or directories containing image files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./output",
        help="Directory where output images will be written",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    logger.info("Starting manga translation pipeline")
    image_paths = collect_image_paths(args.inputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using output directory {}", output_dir)

    image_path_strings = [str(path) for path in image_paths]

    ocr_output, ocr_results = run_ocr(image_path_strings)
    logger.debug("OCR raw output: {}", ocr_output)
    logger.debug("OCR results: {}", ocr_results)

    translations = translate_images(image_path_strings, ocr_results)
    logger.debug("Translations: {}", translations)

    for index, (  # pyright: ignore[reportGeneralTypeIssues]
        image_path,
        image_ocr_output,
        image_ocr_results,
        image_translations,
    ) in enumerate(zip(image_paths, ocr_output, ocr_results, translations), start=1):  # pyright: ignore[reportArgumentType]
        (
            image_output_dir,
            ocr_json_path,
            ocr_markdown_path,
            translation_markdown_path,
            bbox_path,
            text_removed_path,
            translated_path,
        ) = build_output_paths(output_dir, image_path)
        image_output_dir.mkdir(parents=True, exist_ok=True)

        image_ocr_output.save_to_json(ocr_json_path.as_posix())
        logger.debug("Saved OCR JSON: {}", ocr_json_path)

        image_ocr_output.save_to_markdown(ocr_markdown_path.as_posix())
        logger.debug("Saved OCR markdown: {}", ocr_markdown_path)

        save_translations_markdown(
            translation_markdown_path,
            image_ocr_results,
            image_translations,
        )
        logger.debug("Saved translation markdown: {}", translation_markdown_path)

        image = Image.open(image_path)

        all_text_regions = extract_text_regions(image_ocr_output, min_confidence=None)
        bbox_image = draw_ocr_bboxes(
            image,
            all_text_regions,
            confidence_threshold=OCR_CONFIDENCE_THRESHOLD,
        )
        bbox_image.save(bbox_path)
        logger.debug("Saved bbox image: {}", bbox_path)

        cleaned = remove_ocr_regions(image, image_ocr_results)
        cleaned.save(text_removed_path)
        logger.debug("Saved cleaned image: {}", text_removed_path)

        translated_image = draw_translations(cleaned, image_translations)
        translated_image.save(translated_path)
        logger.info(
            "Processed image {} of {}: {} -> {}",
            index,
            len(image_paths),
            image_path,
            translated_path,
        )

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
