import base64
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from loguru import logger

from models import OCRResult, TranslationResult

MAX_TRANSLATION_CONCURRENCY = 32
MAX_COMPLETION_TOKENS = 1024


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


def translate_one_image(
    index: int,
    image_path: str,
    ocr_result: list[OCRResult],
    total_images: int,
    system_prompt: str,
) -> tuple[int, list[TranslationResult]]:
    client = openai.OpenAI(api_key="llama.cpp", base_url="http://localhost:8080/v1")

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
        "Translating image {} of {}: {}", index + 1, total_images, image_path
    )
    logger.debug("Translation request messages: {}", truncate_openai_messages(messages))

    response = client.chat.completions.create(
        model="llama",
        messages=messages,  # ty:ignore[invalid-argument-type]
        max_tokens=MAX_COMPLETION_TOKENS,
    )

    content = response.choices[0].message.content
    if content is None:
        logger.warning(
            "Image {} translation failed: no content in response", index + 1
        )
        return index, []

    output_lines = content.strip().splitlines()
    logger.debug("Image {} translations: {}", index + 1, output_lines)

    if len(output_lines) != len(ocr_result):
        logger.warning(
            "Image {} translation count mismatch: expected {}, got {}",
            index + 1,
            len(ocr_result),
            len(output_lines),
        )

    translations = [
        TranslationResult(ocr_result=res, translation=translation)
        for res, translation in zip(ocr_result, output_lines)
    ]
    logger.debug(
        "Image {}: produced {} translation(s)", index + 1, len(translations)
    )
    return index, translations


def translate_images(
    image_paths: list[str], ocr_results: list[list[OCRResult]]
) -> list[list[TranslationResult]]:
    logger.info("Running translation on {} image(s)", len(image_paths))

    system_prompt = """
    Translate the Japanese text into natural English. The user will provide you with a manga image and the extracted text to translate.
    Translate only the text the user provides. If you detect any text in the image that the user did not send, ignore it.
    Each translation must be on its own line, in the same order as the input. The number of input and output lines must match.

    Example input:
    - text: “そんな傷だらけになってまで、どうして戦おうとするんだ”
    - text: “守りたいものがあるって決めたからだよ”
    - text: “命を落としたら元も子もないだろ”
    - text: “それでも、何もしないまま後悔するのは嫌なんだ”

    Example output:
    Why do you keep fighting when you’re this badly hurt?
    Because I decided there are things worth protecting.
    If you die, none of it will matter.
    Even so, I’d rather risk everything than live with regret.
    """

    translations: list[list[TranslationResult]] = [[] for _ in range(len(image_paths))]
    max_workers = min(len(image_paths), MAX_TRANSLATION_CONCURRENCY)
    logger.info("Using translation concurrency {}", max_workers)

    with ThreadPoolExecutor(max_workers=max_workers or 1) as executor:
        futures = [
            executor.submit(
                translate_one_image,
                i,
                image_path,
                ocr_result,
                len(image_paths),
                system_prompt,
            )
            for i, (image_path, ocr_result) in enumerate(zip(image_paths, ocr_results))
        ]

        for future in as_completed(futures):
            index, image_translations = future.result()
            translations[index] = image_translations

    logger.info("Translation completed")
    return translations
