from paddleocr import PaddleOCRVL

from models import OCRResult


def create_ocr_pipeline() -> PaddleOCRVL:
    return PaddleOCRVL(
        vl_rec_backend="mlx-vlm-server",
        vl_rec_server_url="http://localhost:8111/",
        vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
    )


def parse_ocr_output(ocr_output) -> list[list[OCRResult]]:
    ocr_results: list[list[OCRResult]] = [[] for _ in range(len(ocr_output))]

    for i, res in enumerate(ocr_output):
        for result_block, layout_box in zip(
            res["parsing_res_list"], res["layout_det_res"]["boxes"]
        ):
            label = result_block.label
            confidence = layout_box["score"]

            if "text" not in label or confidence < 0.8:
                continue

            ocr_results[i].append(
                OCRResult(
                    text=result_block.content,
                    bbox=result_block.bbox,
                    confidence=confidence,
                )
            )

    return ocr_results


def run_ocr(image_paths: list[str]) -> list[list[OCRResult]]:
    ocr_pipeline = create_ocr_pipeline()
    ocr_output = ocr_pipeline.predict(image_paths)
    return parse_ocr_output(ocr_output)
