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


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    paragraphs = text.splitlines() or [text]
    wrapped_lines: list[str] = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue

        current_line = words[0]
        for word in words[1:]:
            candidate = f"{current_line} {word}"
            left, top, right, bottom = draw.textbbox((0, 0), candidate, font=font)
            if right - left <= max_width:
                current_line = candidate
                continue

            wrapped_lines.append(current_line)
            current_line = word

            while True:
                left, top, right, bottom = draw.textbbox(
                    (0, 0), current_line, font=font
                )
                if right - left <= max_width or len(current_line) <= 1:
                    break

                split_index = len(current_line) - 1
                while split_index > 1:
                    prefix = current_line[:split_index] + "-"
                    left, top, right, bottom = draw.textbbox((0, 0), prefix, font=font)
                    if right - left <= max_width:
                        wrapped_lines.append(prefix)
                        current_line = current_line[split_index:]
                        break
                    split_index -= 1
                else:
                    break

        while True:
            left, top, right, bottom = draw.textbbox((0, 0), current_line, font=font)
            if right - left <= max_width or len(current_line) <= 1:
                wrapped_lines.append(current_line)
                break

            split_index = len(current_line) - 1
            while split_index > 1:
                prefix = current_line[:split_index] + "-"
                left, top, right, bottom = draw.textbbox((0, 0), prefix, font=font)
                if right - left <= max_width:
                    wrapped_lines.append(prefix)
                    current_line = current_line[split_index:]
                    break
                split_index -= 1
            else:
                wrapped_lines.append(current_line)
                break

    return "\n".join(wrapped_lines)


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

    while low <= high:
        mid = (low + high) // 2
        font = get_font(mid)
        spacing = max(2, mid // 6)
        wrapped_text = wrap_text_to_width(draw, text, font, box_width)
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0), wrapped_text, font=font, spacing=spacing, align="center"
        )
        text_width = right - left
        text_height = bottom - top

        if text_width <= box_width and text_height <= box_height:
            best_text = wrapped_text
            best_font = font
            best_spacing = spacing
            low = mid + 1
        else:
            high = mid - 1

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
