from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from models import BBox, OCRResult, TranslationResult


def get_bbox_rectangle_coords(
    bbox: BBox, image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Convert a bbox into clamped rectangle coordinates for drawing."""
    x1, y1, x2, y2 = bbox
    width, height = image_size

    left = max(0, min(int(x1), int(x2)))
    top = max(0, min(int(y1), int(y2)))
    right = min(width, max(int(x1), int(x2)))
    bottom = min(height, max(int(y1), int(y2)))

    assert left < right, f"Invalid bbox horizontal bounds after clamping: {bbox}"
    assert top < bottom, f"Invalid bbox vertical bounds after clamping: {bbox}"

    return left, top, right, bottom


def remove_ocr_regions(
    image: Image.Image,
    ocr_results: list[OCRResult],
    fill: int | tuple[int, ...] | None = None,
) -> Image.Image:
    image_without_text = image.copy()
    draw = ImageDraw.Draw(image_without_text)

    if fill is None:
        bands = Image.getmodebands(image_without_text.mode)
        fill = (255,) * bands if bands > 1 else 255

    for result in ocr_results:
        rectangle_coords = get_bbox_rectangle_coords(
            result.bbox, image_without_text.size
        )
        draw.rectangle(rectangle_coords, fill=fill)

    return image_without_text


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue

    return ImageFont.load_default()


@dataclass(frozen=True)
class WrappedTextCandidate:
    text: str
    lines: list[str]
    hyphenated_line_count: int
    tiny_fragment_count: int
    overflowed: bool


@dataclass(frozen=True)
class FittedTextCandidate:
    wrapped_text: WrappedTextCandidate
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    spacing: int


def measure_text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
    return right - left


def split_word_to_width(
    draw: ImageDraw.ImageDraw,
    word: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> tuple[list[str], int, int, bool]:
    segments: list[str] = []
    hyphenated_line_count = 0
    tiny_fragment_count = 0
    remaining = word

    while measure_text_width(draw, remaining, font) > max_width and len(remaining) > 1:
        split_index = None
        for min_prefix, min_suffix in ((3, 3), (2, 2), (1, 1)):
            candidate_index = len(remaining) - min_suffix
            while candidate_index >= min_prefix:
                prefix = remaining[:candidate_index] + "-"
                if measure_text_width(draw, prefix, font) <= max_width:
                    split_index = candidate_index
                    if min_prefix == 1 or min_suffix == 1:
                        tiny_fragment_count += 1
                    break
                candidate_index -= 1

            if split_index is not None:
                break

        if split_index is None:
            return [word], hyphenated_line_count, tiny_fragment_count, True

        segments.append(remaining[:split_index] + "-")
        hyphenated_line_count += 1
        remaining = remaining[split_index:]

    segments.append(remaining)
    overflowed = measure_text_width(draw, remaining, font) > max_width
    return segments, hyphenated_line_count, tiny_fragment_count, overflowed


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> WrappedTextCandidate:
    paragraphs = text.splitlines() or [text]
    wrapped_lines: list[str] = []
    hyphenated_line_count = 0
    tiny_fragment_count = 0
    overflowed = False

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue

        current_line = words[0]
        if measure_text_width(draw, current_line, font) > max_width:
            overflowed = True

        for word in words[1:]:
            candidate = f"{current_line} {word}"
            if measure_text_width(draw, candidate, font) <= max_width:
                current_line = candidate
                continue

            wrapped_lines.append(current_line)
            current_line = word
            if measure_text_width(draw, current_line, font) > max_width:
                overflowed = True

        wrapped_lines.append(current_line)

    return WrappedTextCandidate(
        text="\n".join(wrapped_lines),
        lines=wrapped_lines,
        hyphenated_line_count=hyphenated_line_count,
        tiny_fragment_count=tiny_fragment_count,
        overflowed=overflowed,
    )


def wrap_text_with_fallback(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> WrappedTextCandidate:
    natural_wrap = wrap_text_to_width(draw, text, font, max_width)
    if not natural_wrap.overflowed:
        return natural_wrap

    paragraphs = text.splitlines() or [text]
    wrapped_lines: list[str] = []
    hyphenated_line_count = 0
    tiny_fragment_count = 0
    overflowed = False

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue

        current_line = ""
        for word in words:
            candidate = word if not current_line else f"{current_line} {word}"
            if current_line and measure_text_width(draw, candidate, font) <= max_width:
                current_line = candidate
                continue

            if current_line:
                wrapped_lines.append(current_line)
                current_line = ""

            if measure_text_width(draw, word, font) <= max_width:
                current_line = word
                continue

            word_segments, split_count, tiny_count, word_overflowed = split_word_to_width(
                draw, word, font, max_width
            )
            hyphenated_line_count += split_count
            tiny_fragment_count += tiny_count
            overflowed = overflowed or word_overflowed

            wrapped_lines.extend(word_segments[:-1])
            current_line = word_segments[-1]

        if current_line:
            wrapped_lines.append(current_line)

    return WrappedTextCandidate(
        text="\n".join(wrapped_lines),
        lines=wrapped_lines,
        hyphenated_line_count=hyphenated_line_count,
        tiny_fragment_count=tiny_fragment_count,
        overflowed=overflowed,
    )


def score_wrapped_text(
    draw: ImageDraw.ImageDraw,
    wrapped_text: WrappedTextCandidate,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    max_height: int,
    spacing: int,
) -> float:
    non_empty_lines = [line for line in wrapped_text.lines if line]
    if not non_empty_lines:
        return float("-inf")

    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0), wrapped_text.text, font=font, spacing=spacing, align="center"
    )
    text_width = right - left
    text_height = bottom - top
    fill_ratios = [
        measure_text_width(draw, line, font) / max_width for line in non_empty_lines
    ]
    avg_fill_ratio = sum(fill_ratios) / len(fill_ratios)
    min_fill_ratio = min(fill_ratios)
    width_fill_ratio = text_width / max_width
    height_fill_ratio = text_height / max_height
    area_fill_ratio = width_fill_ratio * height_fill_ratio
    short_line_penalty = sum(1 for ratio in fill_ratios if ratio < 0.35)
    single_word_penalty = sum(
        1
        for line in non_empty_lines
        if len(line.rstrip("-").split()) <= 1 and len(non_empty_lines) > 1
    )

    return (
        (font.size * 4)
        + (avg_fill_ratio * 40)
        + (min_fill_ratio * 15)
        + (height_fill_ratio * 120)
        + (area_fill_ratio * 80)
        - (len(non_empty_lines) * 8)
        - (wrapped_text.hyphenated_line_count * 120)
        - (short_line_penalty * 12)
        - (single_word_penalty * 8)
        - (wrapped_text.tiny_fragment_count * 40)
    )


def select_best_fitting_candidate(
    draw: ImageDraw.ImageDraw,
    candidates: list[FittedTextCandidate],
    box_width: int,
    box_height: int,
) -> FittedTextCandidate | None:
    if not candidates:
        return None

    max_font_size = max(candidate.font.size for candidate in candidates)
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate.font.size >= max_font_size * 0.7
    ]
    min_line_count = min(
        len([line for line in candidate.wrapped_text.lines if line])
        for candidate in eligible_candidates
    )
    line_count_candidates = [
        candidate
        for candidate in eligible_candidates
        if len([line for line in candidate.wrapped_text.lines if line]) == min_line_count
    ]

    return max(
        line_count_candidates,
        key=lambda candidate: score_wrapped_text(
            draw,
            candidate.wrapped_text,
            candidate.font,
            box_width,
            box_height,
            candidate.spacing,
        ),
    )


def fit_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box_width: int,
    box_height: int,
) -> tuple[str, ImageFont.FreeTypeFont | ImageFont.ImageFont, int]:
    best_text = text
    best_font = get_font(10)
    best_spacing = 2

    low = 6
    high = max(6, min(box_width, box_height * 2))
    natural_candidates: list[FittedTextCandidate] = []

    for size in range(high, low - 1, -1):
        font = get_font(size)
        spacing = max(2, size // 6)
        wrapped_text = wrap_text_to_width(draw, text, font, box_width)
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0), wrapped_text.text, font=font, spacing=spacing, align="center"
        )
        text_width = right - left
        text_height = bottom - top

        if text_width > box_width or text_height > box_height:
            continue

        natural_candidates.append(
            FittedTextCandidate(
                wrapped_text=wrapped_text,
                font=font,
                spacing=spacing,
            )
        )

    best_candidate = select_best_fitting_candidate(
        draw, natural_candidates, box_width, box_height
    )
    if best_candidate is not None:
        return (
            best_candidate.wrapped_text.text,
            best_candidate.font,
            best_candidate.spacing,
        )

    fallback_candidates: list[FittedTextCandidate] = []
    for size in range(high, low - 1, -1):
        font = get_font(size)
        spacing = max(2, size // 6)
        wrapped_text = wrap_text_with_fallback(draw, text, font, box_width)
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0), wrapped_text.text, font=font, spacing=spacing, align="center"
        )
        text_width = right - left
        text_height = bottom - top

        if text_width > box_width or text_height > box_height:
            continue

        fallback_candidates.append(
            FittedTextCandidate(
                wrapped_text=wrapped_text,
                font=font,
                spacing=spacing,
            )
        )

    best_candidate = select_best_fitting_candidate(
        draw, fallback_candidates, box_width, box_height
    )
    if best_candidate is not None:
        return (
            best_candidate.wrapped_text.text,
            best_candidate.font,
            best_candidate.spacing,
        )

    return best_text, best_font, best_spacing


def draw_ocr_bboxes(
    image: Image.Image,
    ocr_results: list[OCRResult],
    confidence_threshold: float | None = None,
    included_color: str | tuple[int, int, int] = "green",
    excluded_color: str | tuple[int, int, int] = "red",
) -> Image.Image:
    bbox_image = image.copy()
    draw = ImageDraw.Draw(bbox_image)
    font = get_font(14)

    for result in ocr_results:
        left, top, right, bottom = get_bbox_rectangle_coords(result.bbox, bbox_image.size)

        color = included_color
        if confidence_threshold is not None and result.confidence < confidence_threshold:
            color = excluded_color

        draw.rectangle((left, top, right, bottom), outline=color, width=3)

        label = f"{result.confidence:.3f}"
        text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), label, font=font)
        text_width = text_right - text_left
        text_height = text_bottom - text_top

        label_x = left
        label_y = max(0, top - text_height - 6)
        background_coords = (
            label_x - 2,
            label_y - 2,
            label_x + text_width + 2,
            label_y + text_height + 2,
        )
        draw.rectangle(background_coords, fill="white")
        draw.text((label_x, label_y), label, font=font, fill=color)

    return bbox_image


def draw_translations(
    image: Image.Image,
    translations: list[TranslationResult],
    fill: int | tuple[int, ...] = 0,
) -> Image.Image:
    translated_image = image.copy()
    draw = ImageDraw.Draw(translated_image)

    for result in translations:
        left, top, right, bottom = get_bbox_rectangle_coords(
            result.ocr_result.bbox, translated_image.size
        )

        padding_x = max(2, int((right - left) * 0.06))
        padding_y = max(2, int((bottom - top) * 0.06))
        inner_left = left + padding_x
        inner_top = top + padding_y
        inner_right = right - padding_x
        inner_bottom = bottom - padding_y

        if inner_left >= inner_right or inner_top >= inner_bottom:
            inner_left, inner_top, inner_right, inner_bottom = left, top, right, bottom

        wrapped_text, font, spacing = fit_text_to_box(
            draw,
            result.translation.strip(),
            inner_right - inner_left,
            inner_bottom - inner_top,
        )

        text_left, text_top, text_right, text_bottom = draw.multiline_textbbox(
            (0, 0), wrapped_text, font=font, spacing=spacing, align="center"
        )
        text_width = text_right - text_left
        text_height = text_bottom - text_top

        x = inner_left + ((inner_right - inner_left) - text_width) / 2 - text_left
        y = inner_top + ((inner_bottom - inner_top) - text_height) / 2 - text_top

        draw.multiline_text(
            (x, y),
            wrapped_text,
            font=font,
            fill=fill,
            spacing=spacing,
            align="center",
        )

    return translated_image
