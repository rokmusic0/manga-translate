from dataclasses import dataclass

type BBox = tuple[int, int, int, int]


@dataclass
class OCRResult:
    text: str
    bbox: BBox
    confidence: float

    def __str__(self) -> str:
        text = self.text.replace("\n", "")
        return text


@dataclass
class TranslationResult:
    ocr_result: OCRResult
    translation: str
