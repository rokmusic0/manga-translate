import base64
import copy

import openai
from loguru import logger

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
    logger.info("Running translation on {} image(s)", len(image_paths))
    client = openai.OpenAI(api_key="llama.cpp", base_url="http://localhost:8080/v1")

    system_prompt = """
    Translate the Japanese text on the manga page into English to sound as natural as possible.
    The user will provide you with the original manga page image, as well as the OCR results.
    Each translation should be on a separate line and in the same order as the OCR results.

    Example input:
    - text: "こんにちは"
    - text: "世界"
    - text: "今日は美しい日です"

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

        logger.debug(
            "Translating image {} of {}: {}", i + 1, len(image_paths), image_path
        )
        logger.debug(
            "Translation request messages: {}", truncate_openai_messages(messages)
        )

        response = client.chat.completions.create(
            model="llama",
            messages=messages,  # ty:ignore[invalid-argument-type]
        )

        content = response.choices[0].message.content
        if content is None:
            logger.warning("Image {} translation failed: no content in response", i)
            continue

        output_lines = content.strip().split("\n")
        logger.debug("Image {} translations: {}", i, output_lines)

        if len(output_lines) != len(ocr_result):
            logger.warning(
                "Image {} translation count mismatch: expected {}, got {}",
                i,
                len(ocr_result),
                len(output_lines),
            )

        translations[i] = [
            TranslationResult(ocr_result=res, translation=translation)
            for res, translation in zip(ocr_result, output_lines)
        ]
        logger.debug(
            "Image {}: produced {} translation(s)", i + 1, len(translations[i])
        )

    logger.info("Translation completed")
    return translations
