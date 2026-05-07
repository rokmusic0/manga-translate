import base64
import copy

import openai

from models import OCRResult, TranslationResult


def image_to_base64_data_uri(file_path: str) -> str:
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


def truncate_openai_messages(messages: list[dict]) -> list[dict]:
    """Truncate image data in messages for debug printing."""
    messages_copy = copy.deepcopy(messages)

    for message in messages_copy:
        content = message["content"]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "image_url" in item:
                    item["image_url"]["url"] = "[IMAGE DATA URI TRUNCATED]"

    return messages_copy


def translate_images(
    image_paths: list[str], ocr_results: list[list[OCRResult]]
) -> list[list[TranslationResult]]:
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
        input_lines = "\n".join(str(res) for res in ocr_result)
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

        if len(output_lines) != len(ocr_result):
            print(
                f"Image {i} translation count mismatch: expected {len(ocr_result)}, got {len(output_lines)}"
            )

        translations[i] = [
            TranslationResult(ocr_result=res, translation=translation)
            for res, translation in zip(ocr_result, output_lines)
        ]

    return translations
