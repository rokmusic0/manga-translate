from PIL import Image

from edit_image import draw_translations, remove_ocr_regions
from ocr import run_ocr
from translate import translate_images

IMAGE_PATHS = ["./images/yatsuba.png"]
TEXT_REMOVED_OUTPUT_PATH = "./output/yatsuba_ocr_regions_removed.png"
TRANSLATED_OUTPUT_PATH = "./output/yatsuba_translated.png"


def main() -> None:
    ocr_results = run_ocr(IMAGE_PATHS)
    print(ocr_results)

    translations = translate_images(IMAGE_PATHS, ocr_results)
    print(translations)

    image = Image.open(IMAGE_PATHS[0])
    cleaned = remove_ocr_regions(image, ocr_results[0])
    cleaned.save(TEXT_REMOVED_OUTPUT_PATH)

    translated_image = draw_translations(cleaned, translations[0])
    translated_image.save(TRANSLATED_OUTPUT_PATH)


if __name__ == "__main__":
    main()
