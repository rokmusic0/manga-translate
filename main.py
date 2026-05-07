import base64
import copy
from dataclasses import dataclass

import openai
from paddleocr import PaddleOCRVL
from PIL import Image, ImageDraw


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


if __name__ == "__main__":
    main()
