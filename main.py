import base64
import copy
from dataclasses import dataclass

import openai
from paddleocr import PaddleOCRVL
from PIL import Image, ImageDraw, ImageFont


@dataclass
class OCRResult:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float

    def __str__(self) -> str:
        return f'- text: "{self.text}", bbox: {self.bbox}'


@dataclass
class TranslationResult:
    ocr_result: OCRResult
    translation: str


def image_to_base64_data_uri(file_path: str) -> str:
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


def parse_ocr_output(ocr_output) -> list[list[OCRResult]]:
    # A list of OCR results for each image, i.e. ocr_results[i] is the list of OCR results for image i.
    ocr_results: list[list[OCRResult]] = [[] for _ in range(len(ocr_output))]

    for i, res in enumerate(ocr_output):
        for result_block, layout_box in zip(
            res["parsing_res_list"], res["layout_det_res"]["boxes"]
        ):
            label = result_block.label
            confidence = layout_box["score"]

            if "text" not in label or confidence < 0.8:
                continue

            ocr_result = OCRResult(
                text=result_block.content,
                bbox=result_block.bbox,
                confidence=confidence,
            )
            ocr_results[i].append(ocr_result)

    return ocr_results


def get_bbox_rectangle_coords(
    bbox: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Convert a bbox into clamped rectangle coordinates for drawing."""
    x1, y1, x2, y2 = bbox
    width, height = image_size

    left = max(0, min(int(x1), int(x2)))
    top = max(0, min(int(y1), int(y2)))
    right = min(width, max(int(x1), int(x2)))
    bottom = min(height, max(int(y1), int(y2)))

    assert left < right, f"Invalid bbox horizontal bounds after clamping: {bbox}"
    assert top < bottom, f"Invalid bbox vertical bounds after clamping: {bbox}"

    return left, top, right, bottom


def remove_ocr_regions(
    image: Image.Image,
    ocr_results: list[OCRResult],
    fill: int | tuple[int, ...] | None = None,
) -> Image.Image:
    """Return a copy of `image` with each OCR bounding box painted over.

    Args:
        image: Source PIL image.
        ocr_results: OCR detections whose ``bbox`` values are painted over.
        fill: Fill color used to replace the OCR regions. Defaults to opaque white
            in a format appropriate for the image mode.
    """
    image_without_text = image.copy()
    draw = ImageDraw.Draw(image_without_text)

    if fill is None:
        bands = Image.getmodebands(image_without_text.mode)
        fill = (255,) * bands if bands > 1 else 255

    for result in ocr_results:
        rectangle_coords = get_bbox_rectangle_coords(
            result.bbox, image_without_text.size
        )
        draw.rectangle(rectangle_coords, fill=fill)

    return image_without_text


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue

    return ImageFont.load_default()


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    paragraphs = text.splitlines() or [text]
    wrapped_lines: list[str] = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue

        current_line = words[0]
        for word in words[1:]:
            candidate = f"{current_line} {word}"
            left, top, right, bottom = draw.textbbox((0, 0), candidate, font=font)
            if right - left <= max_width:
                current_line = candidate
                continue

            wrapped_lines.append(current_line)
            current_line = word

            while True:
                left, top, right, bottom = draw.textbbox((0, 0), current_line, font=font)
                if right - left <= max_width or len(current_line) <= 1:
                    break

                split_index = len(current_line) - 1
                while split_index > 1:
                    prefix = current_line[:split_index] + "-"
                    left, top, right, bottom = draw.textbbox((0, 0), prefix, font=font)
                    if right - left <= max_width:
                        wrapped_lines.append(prefix)
                        current_line = current_line[split_index:]
                        break
                    split_index -= 1
                else:
                    break

        while True:
            left, top, right, bottom = draw.textbbox((0, 0), current_line, font=font)
            if right - left <= max_width or len(current_line) <= 1:
                wrapped_lines.append(current_line)
                break

            split_index = len(current_line) - 1
            while split_index > 1:
                prefix = current_line[:split_index] + "-"
                left, top, right, bottom = draw.textbbox((0, 0), prefix, font=font)
                if right - left <= max_width:
                    wrapped_lines.append(prefix)
                    current_line = current_line[split_index:]
                    break
                split_index -= 1
            else:
                wrapped_lines.append(current_line)
                break

    return "\n".join(wrapped_lines)


def fit_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_width: int,
    box_height: int,
) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int]:
    best_text = text
    best_font = get_font(10)
    best_spacing = 2

    low = 6
    high = max(6, min(box_width, box_height * 2))

    while low <= high:
        mid = (low + high) // 2
        font = get_font(mid)
        spacing = max(2, mid // 6)
        wrapped_text = wrap_text_to_width(draw, text, font, box_width)
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0), wrapped_text, font=font, spacing=spacing, align="center"
        )
        text_width = right - left
        text_height = bottom - top

        if text_width <= box_width and text_height <= box_height:
            best_text = wrapped_text
            best_font = font
            best_spacing = spacing
            low = mid + 1
        else:
            high = mid - 1

    return best_text, best_font, best_spacing


def draw_translations(
    image: Image.Image,
    translations: list[TranslationResult],
    fill: int | tuple[int, ...] = 0,
) -> Image.Image:
    translated_image = image.copy()
    draw = ImageDraw.Draw(translated_image)

    for result in translations:
        left, top, right, bottom = get_bbox_rectangle_coords(
            result.ocr_result.bbox, translated_image.size
        )

        padding_x = max(2, int((right - left) * 0.06))
        padding_y = max(2, int((bottom - top) * 0.06))
        inner_left = left + padding_x
        inner_top = top + padding_y
        inner_right = right - padding_x
        inner_bottom = bottom - padding_y

        if inner_left >= inner_right or inner_top >= inner_bottom:
            inner_left, inner_top, inner_right, inner_bottom = left, top, right, bottom

        wrapped_text, font, spacing = fit_text_to_box(
            draw,
            result.translation.strip(),
            inner_right - inner_left,
            inner_bottom - inner_top,
        )

        text_left, text_top, text_right, text_bottom = draw.multiline_textbbox(
            (0, 0), wrapped_text, font=font, spacing=spacing, align="center"
        )
        text_width = text_right - text_left
        text_height = text_bottom - text_top

        x = inner_left + ((inner_right - inner_left) - text_width) / 2 - text_left
        y = inner_top + ((inner_bottom - inner_top) - text_height) / 2 - text_top

        draw.multiline_text(
            (x, y),
            wrapped_text,
            font=font,
            fill=fill,
            spacing=spacing,
            align="center",
        )

    return translated_image


def truncate_openai_messages(messages: list[dict]) -> list[dict]:
    """Truncate the image data in the messages to avoid printing the whole base64 data.

    This should not be used to send messages to the API, only for debugging, to avoid printing the whole base64 data in the console, as it takes a lot of space.
    """
    messages_copy = copy.deepcopy(messages)

    for message in messages_copy:
        content = message["content"]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "image_url" in item:
                    item["image_url"]["url"] = "[IMAGE DATA URI TRUNCATED]"

    return messages_copy


def translate(image_paths: list[str], ocr_results: list[list[OCRResult]]):
    client = openai.OpenAI(api_key="llama.cpp", base_url="http://localhost:8080/v1")

    system_prompt = """
    I want to translate a Japanese manga page into English.
    I have already extracted the Japanese text and its bounding boxes from the manga page using SOTA OCR model.
    I will give the OCR results, as well as the original manga page image, and want you to translate the Japanese text into English.
    The image is included for reference, for you to be able to understand the context of the text, and to make sure the translation is accurate and natural.

    You may assume the OCR results are accurate, but may not appear in the correct reading order. You should reference the image to understand the correct reading order.
    However, the bounding boxes are accurate, so you can use them to determine the reading order of the text, or the layout of the text in the manga page. (Or not, whatever you think is best to produce the best translation.)
    Note: the reading order and image are there only to make the translations better. You could translate each line independently, but should translate each line in the context of the whole manga page (the image and the other lines of text), to produce better translations.

    The translation should be line by line (i.e. each line of Japanese should be translated into its own line of English), and the order of the lines in the output should be the same as the order of input lines (i.e. the order of the OCR results, the input order). The output should contain English translations only, one per line, and nothing else. I will parse the output programatically, so it is assumed `len(input_lines) == len(output_lines)`, and `output_lines[i]` is the translation of `input_lines[i]`.

    Example input:
    (The image will be included. The OCR results will be given as follows.)
    - text: "こんにちは", bbox: (100, 100, 200, 150)
    - text: "世界", bbox: (300, 100, 400, 150)
    - text: "今日は美しい日です", bbox: (400, 150, 500, 250)

    Example output:
    Hello.
    World.
    Today is a beautiful day.
    """

    translations: list[list[TranslationResult]] = [[] for _ in range(len(image_paths))]

    for i, (ocr_result, image_path) in enumerate(zip(ocr_results, image_paths)):
        input_lines = "\n".join([str(res) for res in ocr_result])
        image_data_uri = image_to_base64_data_uri(image_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri, "detail": "high"},
                    },
                    {"type": "text", "text": input_lines},
                ],
            },
        ]

        print(60 * "=")
        print(truncate_openai_messages(messages))
        print(60 * "=")

        response = client.chat.completions.create(
            model="llama",
            messages=messages,  # ty:ignore[invalid-argument-type]
        )

        content = response.choices[0].message.content
        if content is None:
            print(f"Image {i} translation failed: no content in response")
            continue

        output_lines = content.strip().split("\n")
        print(f"Image {i} translations:")
        for line in output_lines:
            print(line)

        translations[i] = [
            TranslationResult(ocr_result=res, translation=translation)
            for res, translation in zip(ocr_result, output_lines)
        ]

        # break after first image for testing
        break

    return translations


def main():
    image_paths = ["./images/yatsuba.png"]
    # image_paths = ["./images/yatsuba.png", "./images/yatsuba.png", "./images/yatsuba.png"]

    ocr_pipeline = PaddleOCRVL(
        vl_rec_backend="mlx-vlm-server",
        vl_rec_server_url="http://localhost:8111/",
        vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
    )
    ocr_output = ocr_pipeline.predict(image_paths)
    ocr_results = parse_ocr_output(ocr_output)
    print(ocr_results)

    translations = translate(image_paths, ocr_results)
    print(translations)

    image = Image.open("./images/yatsuba.png")
    cleaned = remove_ocr_regions(image, ocr_results[0])
    cleaned.save("./output/text_removed.png")

    translated_image = draw_translations(cleaned, translations[0])
    translated_image.save("./output/yatsuba_translated.png")


if __name__ == "__main__":
    main()
