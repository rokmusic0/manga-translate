from paddleocr import PaddleOCRVL
from llama_cpp import Llama
import base64
from dataclasses import dataclass


@dataclass
class OCRResult:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float


def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


image_paths = ["./images/yatsuba.png", "./images/yatsuba.png", "./images/yatsuba.png"]

ocr_pipeline = PaddleOCRVL(
    vl_rec_backend="mlx-vlm-server",
    vl_rec_server_url="http://localhost:8111/",
    vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
)
ocr_output = ocr_pipeline.predict(image_paths)


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


ocr_results = parse_ocr_output(ocr_output)

print(ocr_results)
