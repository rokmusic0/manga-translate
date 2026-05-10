from loguru import logger
from paddleocr import PaddleOCRVL

from models import OCRResult


def create_ocr_pipeline() -> PaddleOCRVL:
    logger.debug("Creating OCR pipeline")
    return PaddleOCRVL(
        vl_rec_backend="mlx-vlm-server",
        vl_rec_server_url="http://localhost:8111/",
        vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        vl_rec_max_concurrency=1,
        use_queues=False,
    )


def parse_ocr_output(ocr_output) -> list[list[OCRResult]]:
    logger.debug("Parsing OCR output for {} image(s)", len(ocr_output))
    ocr_results: list[list[OCRResult]] = [[] for _ in range(len(ocr_output))]

    for i, res in enumerate(ocr_output):
        for result_block, layout_box in zip(
            res["parsing_res_list"], res["layout_det_res"]["boxes"]
        ):
            label = result_block.label
            confidence = layout_box["score"]

            if "text" not in label or confidence < 0.5:
                continue

            ocr_results[i].append(
                OCRResult(
                    text=result_block.content,
                    bbox=result_block.bbox,
                    confidence=confidence,
                )
            )

        logger.debug(
            "Image {}: extracted {} text region(s)", i + 1, len(ocr_results[i])
        )

    logger.debug("Parsed OCR results: {}", ocr_results)
    return ocr_results


def run_ocr(image_paths: list[str]) -> list[list[OCRResult]]:
    logger.info("Running OCR on {} image(s)", len(image_paths))
    logger.debug("OCR input paths: {}", image_paths)
    ocr_pipeline = create_ocr_pipeline()
    ocr_output = ocr_pipeline.predict(image_paths)
    logger.info("OCR completed")
    return parse_ocr_output(ocr_output)
