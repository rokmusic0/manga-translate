import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from loguru import logger

from models import OCRResult, TranslationResult

MAX_TRANSLATION_CONCURRENCY = 32
MAX_COMPLETION_TOKENS = 1024
LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080/v1")
LLAMA_MODEL_NAME = os.environ.get("LLAMA_MODEL_NAME", "llama")


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
    if not ocr_result:
        logger.warning("Image {} has no OCR results, skipping translation", index + 1)
        return index, []

    client = openai.OpenAI(api_key="llama.cpp", base_url=LLAMA_BASE_URL)

    input_lines = "\n".join(str(res) for res in ocr_result)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_lines},
    ]

    logger.debug("Translating image {} of {}: {}", index + 1, total_images, image_path)
    logger.debug("Translation request messages: {}", truncate_openai_messages(messages))

    response = client.chat.completions.create(
        model=LLAMA_MODEL_NAME,
        messages=messages,  # ty:ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
        temperature=0.0,
        max_tokens=MAX_COMPLETION_TOKENS,
    )

    content = response.choices[0].message.content
    if content is None:
        logger.warning("Image {} translation failed: no content in response", index + 1)
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
    logger.debug("Image {}: produced {} translation(s)", index + 1, len(translations))
    return index, translations


def translate_images(
    image_paths: list[str], ocr_results: list[list[OCRResult]]
) -> list[list[TranslationResult]]:
    logger.info("Running translation on {} image(s)", len(image_paths))

    system_prompt = """
    Translate the Japanese text into natural English.
    Each translation must be on its own line, in the same order as the input. The number of input and output lines must match.
    This includes translating any sound effects or onomatopoeia into English as well, to the best of your ability, even lines like "...".

    Example input:
    そんな傷だらけになってまで、どうして戦おうとするんだ
    守りたいものがあるって決めたからだよ
    ...
    命を落としたら元も子もないだろ
    それでも、何もしないまま後悔するのは嫌なんだ

    Example output:
    Why do you keep fighting when you're this badly hurt?
    Because I decided there are things worth protecting.
    ...
    If you die, none of it will matter.
    Even so, I'd rather risk everything than live with regret.
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
