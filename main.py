import argparse
from pathlib import Path

from PIL import Image

from edit_image import draw_translations, remove_ocr_regions
from ocr import run_ocr
from translate import translate_images

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_paths(inputs: list[str]) -> list[Path]:
    image_paths: list[Path] = []

    for input_path in inputs:
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_dir():
            image_paths.extend(
                sorted(file for file in path.iterdir() if is_image_file(file))
            )
        elif is_image_file(path):
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

    return deduped_paths


def build_output_paths(output_dir: Path, image_path: Path) -> tuple[Path, Path]:
    text_removed_path = output_dir / f"{image_path.stem}_ocr_regions_removed.png"
    translated_path = output_dir / f"{image_path.stem}_translated.png"
    return text_removed_path, translated_path


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = collect_image_paths(args.inputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path_strings = [str(path) for path in image_paths]

    ocr_results = run_ocr(image_path_strings)
    print(ocr_results)

    translations = translate_images(image_path_strings, ocr_results)
    print(translations)

    for image_path, image_ocr_results, image_translations in zip(
        image_paths, ocr_results, translations
    ):
        text_removed_path, translated_path = build_output_paths(output_dir, image_path)

        image = Image.open(image_path)
        cleaned = remove_ocr_regions(image, image_ocr_results)
        cleaned.save(text_removed_path)

        translated_image = draw_translations(cleaned, image_translations)
        translated_image.save(translated_path)

        print(f"Saved cleaned image: {text_removed_path}")
        print(f"Saved translated image: {translated_path}")


if __name__ == "__main__":
    main()
