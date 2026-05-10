from loguru import logger
from paddleocr import PaddleOCRVL

from models import OCRResult

OCR_CONFIDENCE_THRESHOLD = 0.5


def create_ocr_pipeline() -> PaddleOCRVL:
    logger.debug("Creating OCR pipeline")
    return PaddleOCRVL(
        vl_rec_backend="mlx-vlm-server",
        vl_rec_server_url="http://localhost:8111/",
        vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        vl_rec_max_concurrency=1,
        use_queues=False,
    )


def extract_text_regions(
    ocr_page_output, min_confidence: float | None = OCR_CONFIDENCE_THRESHOLD
) -> list[OCRResult]:
    results: list[OCRResult] = []

    for result_block, layout_box in zip(
        ocr_page_output["parsing_res_list"], ocr_page_output["layout_det_res"]["boxes"]
    ):
        label = result_block.label
        confidence = layout_box["score"]

        if "text" not in label:
            continue

        if min_confidence is not None and confidence < min_confidence:
            continue

        results.append(
            OCRResult(
                text=result_block.content,
                bbox=result_block.bbox,
                confidence=confidence,
            )
        )

    return results


def parse_ocr_output(ocr_output) -> list[list[OCRResult]]:
    logger.debug("Parsing OCR output for {} image(s)", len(ocr_output))
    ocr_results = [extract_text_regions(res) for res in ocr_output]

    for i, image_results in enumerate(ocr_results):
        logger.debug("Image {}: extracted {} text region(s)", i + 1, len(image_results))

    logger.debug("Parsed OCR results: {}", ocr_results)
    return ocr_results


def run_ocr(image_paths: list[str]) -> tuple[object, list[list[OCRResult]]]:
    logger.info("Running OCR on {} image(s)", len(image_paths))
    logger.debug("OCR input paths: {}", image_paths)
    ocr_pipeline = create_ocr_pipeline()
    ocr_output = ocr_pipeline.predict(image_paths)
    return ocr_output, parse_ocr_output(ocr_output)
