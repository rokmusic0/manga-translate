from dataclasses import dataclass

type BBox = tuple[int, int, int, int]


@dataclass
class OCRResult:
    text: str
    bbox: BBox
    confidence: float

    def __str__(self) -> str:
        return f'- text: "{self.text}", bbox: {self.bbox}'


@dataclass
class TranslationResult:
    ocr_result: OCRResult
    translation: str
