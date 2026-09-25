import asyncio
import gc
import hashlib
import io
import json
import os
import time
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import flet as ft
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import arabic_reshaper
from bidi.algorithm import get_display

# حذف زمینه با ONNX Runtime و مدل سبک U²-NetP.
# مدل در اولین استفاده به‌صورت خودکار داخل storage خصوصی برنامه دانلود و cache می‌شود.
try:
    import numpy as np
    import onnxruntime as ort

    BACKGROUND_REMOVAL_AVAILABLE = True
except Exception:
    np = None
    ort = None
    BACKGROUND_REMOVAL_AVAILABLE = False

BACKGROUND_MODEL_FILENAME = "u2netp.onnx"
BACKGROUND_MODEL_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
BACKGROUND_MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
BACKGROUND_MODEL_DOWNLOAD_CHUNK = 128 * 1024
SEGMENT_INPUT_SIZE = (320, 320)
SEGMENT_MAX_SIDE = 1280

# ============================================================
# تنظیمات قالب ۸۴۰×۱۲۶۰ — مختصات در همهٔ زمینه‌ها مشترک است
# ============================================================
BOX_TOP_NUMBER = {"x": 420, "y": 190, "w": 300, "h": 60}
BOX_TOP_TITLE = {"x": 420, "y": 290, "w": 400, "h": 85}
BOX_MIDDLE = {"x": 420, "y": 605, "w": 470, "h": 135}
BOX_PHOTO = {"x": 275, "y": 875, "w": 183, "h": 244}
BOX_PRESENTER = {"x": 490, "y": 915, "w": 225, "h": 90}
BOX_DATE = {"x": 410, "y": 1080, "w": 205, "h": 35}

TEMPLATE_SIZE = (853, 1280)
# زمینه‌های قابل انتخاب؛ همهٔ آن‌ها از مختصات مشترک متن و عکس استفاده می‌کنند.
TEMPLATE_FILES = tuple(
    (f"قالب شماره {index}", f"template_{index}.jpg")
    for index in range(1, 8)
)
TEMPLATE_COUNT = len(TEMPLATE_FILES)

# ============================================================
# رنگ‌ها
# ============================================================
BG = "#071713"
PANEL = "#10231E"
PANEL_2 = "#172F29"
GOLD = "#D4AF37"
CREAM = "#FDF6E3"
NAVY = "#1A2E2A"
MUTED = "#91A39C"
DANGER = "#D96C6C"
SUCCESS = "#86C99A"
BORDER = "#29463E"
BLACK_GLASS = "#C0071713"

WEEKDAYS = [
    "دوشنبه",
    "سه‌شنبه",
    "چهارشنبه",
    "پنجشنبه",
    "جمعه",
    "شنبه",
    "یکشنبه",
]
JALALI_MONTHS = [
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
]
PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
INITIAL_SESSION_NUMBER = 161
SESSION_MIN = 161
SESSION_MAX = 300
APP_VERSION = "2.0 Professional"

# ============================================================
# اعداد ترتیبی جلسات ۱۵۰ تا ۳۰۰
# ============================================================
ONES_CARDINAL = {
    1: "یک",
    2: "دو",
    3: "سه",
    4: "چهار",
    5: "پنج",
    6: "شش",
    7: "هفت",
    8: "هشت",
    9: "نه",
}
ONES_ORDINAL = {
    1: "یکمین",
    2: "دومین",
    3: "سومین",
    4: "چهارمین",
    5: "پنجمین",
    6: "ششمین",
    7: "هفتمین",
    8: "هشتمین",
    9: "نهمین",
}
TEENS_ORDINAL = {
    10: "دهمین",
    11: "یازدهمین",
    12: "دوازدهمین",
    13: "سیزدهمین",
    14: "چهاردهمین",
    15: "پانزدهمین",
    16: "شانزدهمین",
    17: "هفدهمین",
    18: "هجدهمین",
    19: "نوزدهمین",
}
TENS_CARDINAL = {
    20: "بیست",
    30: "سی",
    40: "چهل",
    50: "پنجاه",
    60: "شصت",
    70: "هفتاد",
    80: "هشتاد",
    90: "نود",
}
TENS_ORDINAL = {
    20: "بیستمین",
    30: "سی‌امین",
    40: "چهلمین",
    50: "پنجاهمین",
    60: "شصتمین",
    70: "هفتادمین",
    80: "هشتادمین",
    90: "نودمین",
}


def ordinal_under_100(n: int) -> str:
    if n in ONES_ORDINAL:
        return ONES_ORDINAL[n]
    if n in TEENS_ORDINAL:
        return TEENS_ORDINAL[n]
    tens = (n // 10) * 10
    unit = n % 10
    if unit == 0:
        return TENS_ORDINAL[tens]
    return f"{TENS_CARDINAL[tens]} و {ONES_ORDINAL[unit]}"


def ordinal_persian(n: int) -> str:
    """تولید متن ترتیبی فارسی برای ۱۶۱ تا ۳۰۰."""
    if n < 161 or n > 300:
        raise ValueError("شماره جلسه باید بین ۱۶۱ تا ۳۰۰ باشد.")

    if n == 200:
        return "دویستمین"
    if n == 300:
        return "سیصدمین"

    hundreds = n // 100
    remainder = n % 100

    hundred_words = {
        1: "صد",
        2: "دویست",
        3: "سیصد",
    }

    prefix = hundred_words[hundreds]

    if remainder == 0:
        return {
            1: "صدمین",
            2: "دویستمین",
            3: "سیصدمین",
        }[hundreds]

    return f"{prefix} و {ordinal_under_100(remainder)}"

# متن‌های ترتیبی به صورت صریح در حافظه برنامه تعریف می‌شوند.
SESSION_ORDINALS = {n: ordinal_persian(n) for n in range(SESSION_MIN, SESSION_MAX + 1)}
SESSION_ORDINAL_TO_NUMBER = {v: k for k, v in SESSION_ORDINALS.items()}

# ============================================================
# تاریخ شمسی
# ============================================================
def gregorian_to_jalali(gy: int, gm: int, gd: int):
    g_days_in_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    if gy > 1600:
        jy = 979
        gy2 = gy - 1600
    else:
        jy = 0
        gy2 = gy - 621

    gy3 = gy2 + 1 if gm > 2 else gy2
    days = (
        365 * gy2
        + (gy3 + 3) // 4
        - (gy3 + 99) // 100
        + (gy3 + 399) // 400
        - 80
        + gd
        + g_days_in_month[gm - 1]
    )

    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461

    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365

    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30

    return jy, jm, jd


def to_persian_digits(value) -> str:
    return str(value).translate(PERSIAN_DIGITS)


def today_jalali_text() -> str:
    now = date.today()
    jy, jm, jd = gregorian_to_jalali(now.year, now.month, now.day)
    weekday = WEEKDAYS[now.weekday()]
    return (
        f"{weekday} {to_persian_digits(jd)} "
        f"{JALALI_MONTHS[jm - 1]} {to_persian_digits(jy)}"
    )

# ============================================================
# پردازش فارسی
# ============================================================
def prepare_farsi_text(text: str) -> str:
    if not text:
        return ""
    return get_display(arabic_reshaper.reshape(text))


def draw_centered_text(draw, text, box, font, fill_color):
    if not text:
        return

    shaped = prepare_farsi_text(text)
    bbox = draw.textbbox((0, 0), shaped, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x = box["x"] - width / 2
    y = box["y"] - height / 2
    draw.text((x, y), shaped, font=font, fill=fill_color)


def load_font(font_path: str, size: int):
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


# ============================================================
# اندازه‌گذاری هوشمند متن در کادر
# ============================================================
SUBJECT_BASE_FONT_SIZE = 72
SUBJECT_BASE_LINE_SPACING = 72
PRESENTER_BASE_FONT_SIZE = 42
PRESENTER_BASE_LINE_SPACING = 52

AUTO_FIT_HORIZONTAL_PADDING = 6
AUTO_FIT_VERTICAL_PADDING = 4
AUTO_FIT_MIN_FONT_SIZE = 10


def _measure_shaped_line(draw, text: str, font):
    shaped = prepare_farsi_text(text)
    if not shaped:
        return shaped, 0, 0
    bbox = draw.textbbox((0, 0), shaped, font=font)
    width = max(0, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])
    return shaped, width, height


def fit_font_size_for_lines(
    draw,
    lines,
    box,
    font_path: str,
    base_size: int,
    base_line_spacing: int,
    min_size: int = AUTO_FIT_MIN_FONT_SIZE,
):
    """
    از اندازهٔ فونت بزرگ شروع می‌کند و تا جایی که کل متن دقیقاً داخل کادر
    جا شود، اندازهٔ فونت را کاهش می‌دهد.
    """
    normalized_lines = [str(line or "") for line in (lines or [""])]
    available_width = max(1, int(box["w"] - 2 * AUTO_FIT_HORIZONTAL_PADDING))
    available_height = max(1, int(box["h"] - 2 * AUTO_FIT_VERTICAL_PADDING))

    size = int(base_size)
    min_size = max(1, int(min_size))

    while size >= min_size:
        font = load_font(font_path, size)
        measured = [_measure_shaped_line(draw, line, font) for line in normalized_lines]
        widths = [item[1] for item in measured]
        heights = [item[2] for item in measured]

        max_width = max(widths, default=0)
        if len(normalized_lines) <= 1:
            line_spacing = 0
            total_height = max(heights, default=0)
        else:
            line_spacing = max(1, int(round(base_line_spacing * size / base_size)))
            total_height = sum(heights) + line_spacing * (len(normalized_lines) - 1)

        if max_width <= available_width and total_height <= available_height:
            return font, line_spacing

        size -= 1

    # برای متن‌های فوق‌العاده طولانی، حداقل اندازهٔ تعیین‌شده حفظ می‌شود.
    font = load_font(font_path, min_size)
    if len(normalized_lines) <= 1:
        return font, 0
    return font, max(1, int(round(base_line_spacing * min_size / base_size)))

def draw_fitted_centered_lines(
    draw,
    lines,
    box,
    font_path: str,
    base_size: int,
    base_line_spacing: int,
    fill_color,
):
    """رسم متن چندخطی با بزرگ‌ترین فونت ممکن و بدون خروج از کادر."""
    normalized_lines = [str(line or "") for line in (lines or [""])]
    font, line_spacing = fit_font_size_for_lines(
        draw,
        normalized_lines,
        box,
        font_path,
        base_size,
        base_line_spacing,
    )

    measured = [_measure_shaped_line(draw, line, font) for line in normalized_lines]
    heights = [item[2] for item in measured]

    if len(normalized_lines) <= 1:
        draw_centered_text(draw, normalized_lines[0], box, font, fill_color)
        return font

    total_height = sum(heights) + line_spacing * (len(normalized_lines) - 1)
    cursor_top = box["y"] - total_height / 2

    for index, line in enumerate(normalized_lines):
        line_height = heights[index]
        line_box = box.copy()
        line_box["y"] = cursor_top + line_height / 2
        draw_centered_text(draw, line, line_box, font, fill_color)
        cursor_top += line_height + line_spacing

    return font

# ============================================================
# عکس — فقط تصویر ثابت
# ============================================================
def normalize_image_bytes(value) -> bytes:
    if value is None:
        raise ValueError("اطلاعات عکس خالی است.")

    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, bytearray):
        raw = bytes(value)
    elif isinstance(value, memoryview):
        raw = value.tobytes()
    elif hasattr(value, "getvalue"):
        raw = value.getvalue()
    elif hasattr(value, "read"):
        try:
            old_pos = value.tell() if hasattr(value, "tell") else 0
            if hasattr(value, "seek"):
                value.seek(0)
            raw = value.read()
            if hasattr(value, "seek"):
                value.seek(old_pos)
        except Exception as exc:
            raise ValueError("خواندن دادهٔ عکس ممکن نشد.") from exc
    else:
        raise TypeError(f"نوع دادهٔ عکس پشتیبانی نمی‌شود: {type(value)!r}")

    if not isinstance(raw, bytes):
        raw = bytes(raw)

    if not raw:
        raise ValueError("فایل عکس خالی است.")

    try:
        with Image.open(io.BytesIO(raw)) as test_image:
            image_format = (test_image.format or "").upper()
            is_animated = bool(getattr(test_image, "is_animated", False))
            test_image.verify()

        if image_format not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(
                "فرمت مجاز نیست. فقط JPG، JPEG، PNG و WEBP قابل انتخاب هستند."
            )
        if is_animated:
            raise ValueError("تصاویر متحرک مانند GIF مجاز نیستند.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("فایل انتخاب‌شده یک تصویر معتبر نیست یا ناقص است.") from exc

    return raw


def open_normalized_image(raw_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw_bytes)) as source:
        image = ImageOps.exif_transpose(source.convert("RGBA"))
        image.load()
    return image

# ============================================================
# ساخت پوستر
# ============================================================
def create_poster(template_path, font_path, output_path, data, photo_bytes=None):
    if not template_path.exists():
        raise FileNotFoundError(f"فایل زمینه پیدا نشد: {template_path.name}")

    if not font_path.exists():
        raise FileNotFoundError(
            "فایل calibrib.ttf پیدا نشد. آن را داخل پوشه assets قرار دهید."
        )

    with Image.open(template_path) as template_check:
        if template_check.size != TEMPLATE_SIZE:
            raise ValueError(
                f"اندازهٔ زمینه باید دقیقاً {TEMPLATE_SIZE[0]}×{TEMPLATE_SIZE[1]} باشد."
            )

    img = Image.open(template_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    font_number = load_font(str(font_path), 43)
    font_title = load_font(str(font_path), 50)
    font_subject = load_font(str(font_path), 55)
    font_presenter = load_font(str(font_path), 26)
    font_date = load_font(str(font_path), 25)

    draw_centered_text(draw, data["number"], BOX_TOP_NUMBER, font_number, GOLD)
    draw_centered_text(draw, data["title"], BOX_TOP_TITLE, font_title, GOLD)

    lines = data["subject"].splitlines() or [""]
    draw_fitted_centered_lines(
        draw,
        lines,
        BOX_MIDDLE,
        str(font_path),
        base_size=SUBJECT_BASE_FONT_SIZE,
        base_line_spacing=SUBJECT_BASE_LINE_SPACING,
        fill_color=NAVY,
    )

    if photo_bytes:
        raw_photo = normalize_image_bytes(photo_bytes)
        photo = open_normalized_image(raw_photo)
        target_w = BOX_PHOTO["w"]
        target_h = BOX_PHOTO["h"]
        photo.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        px = (target_w - photo.width) // 2
        py = (target_h - photo.height) // 2
        canvas.alpha_composite(photo, (px, py))

        paste_x = int(BOX_PHOTO["x"] - target_w / 2)
        paste_y = int(BOX_PHOTO["y"] - target_h / 2)
        img.alpha_composite(canvas, (paste_x, paste_y))

    presenter_lines = data["presenter"].splitlines() or [""]
    draw_fitted_centered_lines(
        draw,
        presenter_lines,
        BOX_PRESENTER,
        str(font_path),
        base_size=PRESENTER_BASE_FONT_SIZE,
        base_line_spacing=PRESENTER_BASE_LINE_SPACING,
        fill_color=NAVY,
    )

    draw_centered_text(draw, data["date"], BOX_DATE, font_date, NAVY)

    img.convert("RGB").save(output_path, "JPEG", quality=95)
    return output_path

# ============================================================
# اجرای برنامه
# ============================================================
def main(page: ft.Page):
    page.title = "پوستر ساز جلسات فقهی"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = BG
    page.padding = 0
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    def get_assets_dir() -> Path:
        default_assets_dir = Path(__file__).resolve().parent / "assets"
        return Path(os.environ.get("FLET_ASSETS_DIR", str(default_assets_dir))).resolve()

    def get_storage_dir() -> Path:
        storage_dir = os.environ.get("FLET_APP_STORAGE_DATA")
        if storage_dir:
            return Path(storage_dir).resolve()
        return Path(__file__).resolve().parent / "app_data"

    assets_dir = get_assets_dir()
    storage_dir = get_storage_dir()
    output_dir = storage_dir / "output"
    draft_photo_path = storage_dir / "draft_photo.png"
    state_path = storage_dir / "app_state.json"
    draft_path = storage_dir / "draft.json"

    storage_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    font_path = assets_dir / "calibrib.ttf"
    template_paths = {
        name: assets_dir / filename for name, filename in TEMPLATE_FILES
    }
    template_bytes = {}
    for name, path in template_paths.items():
        try:
            if path.exists():
                template_bytes[name] = path.read_bytes()
        except OSError:
            template_bytes[name] = None

    file_picker = ft.FilePicker()
    page.services.append(file_picker)
    share_service = ft.Share()

    selected_template = {"name": None, "path": None}
    pending_restore = {"state": None}
    selected_photo_bytes = {"value": None}
    current_poster = {"bytes": None, "path": None}
    bg_session = {"value": None}
    bg_model_dir = storage_dir / "models"
    bg_model_path = bg_model_dir / BACKGROUND_MODEL_FILENAME
    bg_removing = {"value": False}

    # ========================================================
    # وضعیت ذخیره‌شده و پیش‌نویس
    # ========================================================
    def read_json(path: Path, default=None):
        if default is None:
            default = {}
        try:
            if not path.exists():
                return default
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def atomic_write_json(path: Path, data):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    app_state = read_json(
        state_path,
        {"last_session_number": SESSION_MIN - 1, "last_template": None},
    )
    if not isinstance(app_state, dict):
        app_state = {"last_session_number": SESSION_MIN - 1}

    draft_state = read_json(draft_path, {})
    draft_save_task = {"value": None}

    def save_app_state():
        atomic_write_json(state_path, app_state)

    def current_next_session_number():
        last = int(app_state.get("last_session_number", SESSION_MIN - 1) or SESSION_MIN - 1)
        if last < SESSION_MIN - 1:
            last = SESSION_MIN - 1
        if last >= SESSION_MAX:
            return SESSION_MAX
        return last + 1

    def current_next_session_text():
        return SESSION_ORDINALS.get(
            current_next_session_number(),
            SESSION_ORDINALS[SESSION_MAX],
        )

    def extract_session_number(text: str):
        text = (text or "").strip()
        if text in SESSION_ORDINAL_TO_NUMBER:
            return SESSION_ORDINAL_TO_NUMBER[text]
        return None

    # ========================================================
    # فیلدها
    # ========================================================
    def border_style():
        return {
            ft.ControlState.DEFAULT: ft.OutlineInputBorder(
                border_radius=12,
                side=ft.BorderSide(width=1, color=BORDER),
            ),
            ft.ControlState.FOCUSED: ft.OutlineInputBorder(
                border_radius=12,
                side=ft.BorderSide(width=2, color=GOLD),
            ),
        }

    txt_number = ft.TextField(
        label="شماره جلسه",
        value=current_next_session_text(),
        text_align=ft.TextAlign.RIGHT,
        filled=True,
        bgcolor=PANEL_2,
        border=border_style(),
        hint_text="مثلاً: صد و هفتاد و دومین",
    )

    txt_title = ft.TextField(
        label="عنوان جلسه",
        value="جلسه شنبه های فقهی",
        text_align=ft.TextAlign.RIGHT,
        filled=True,
        bgcolor=PANEL_2,
        border=border_style(),
    )

    txt_subject = ft.TextField(
        label="موضوع جلسه (چند خطی)",
        value="",
        multiline=True,
        min_lines=2,
        max_lines=4,
        text_align=ft.TextAlign.RIGHT,
        filled=True,
        bgcolor=PANEL_2,
        border=border_style(),
    )

    txt_presenter = ft.TextField(
        label="نام ارائه‌دهنده (چند خطی)",
        value="",
        multiline=True,
        min_lines=2,
        max_lines=3,
        text_align=ft.TextAlign.RIGHT,
        filled=True,
        bgcolor=PANEL_2,
        border=border_style(),
    )

    txt_date = ft.TextField(
        label="تاریخ برگزاری",
        value=today_jalali_text(),
        text_align=ft.TextAlign.RIGHT,
        filled=True,
        bgcolor=PANEL_2,
        border=border_style(),
    )

    txt_photo_status = ft.Text(
        "عکسی انتخاب نشده — پس از انتخاب، وارد تنظیم عکس می‌شوید.",
        color=MUTED,
        size=11,
    )

    status_text = ft.Text(
        "برای شروع، یک زمینه انتخاب کنید.",
        size=11,
        color=MUTED,
        text_align=ft.TextAlign.CENTER,
    )

    subject_count = ft.Text("۰ نویسه", size=10, color=MUTED)
    presenter_count = ft.Text("۰ نویسه", size=10, color=MUTED)

    # ========================================================
    # کمک‌های UI
    # ========================================================
    page_content = ft.Container(expand=True)

    def card_container(content, padding=14, bgcolor=PANEL, border_color=BORDER, radius=18):
        return ft.Container(
            padding=padding,
            bgcolor=bgcolor,
            border=ft.Border.all(1, border_color),
            border_radius=radius,
            content=content,
        )

    def section_title(icon, title, subtitle=None):
        rows = [
            ft.Row(
                [
                    ft.Container(
                        width=36,
                        height=36,
                        bgcolor=BLACK_GLASS,
                        border_radius=11,
                        alignment=ft.Alignment.CENTER,
                        content=ft.Icon(icon, color=GOLD, size=20),
                    ),
                    ft.Column(
                        [
                            ft.Text(
                                title,
                                size=18,
                                weight=ft.FontWeight.BOLD,
                                color=CREAM,
                            ),
                            ft.Text(
                                subtitle,
                                size=10,
                                color=MUTED,
                            ) if subtitle else ft.Container(),
                        ],
                        spacing=1,
                        expand=True,
                    ),
                ],
                spacing=10,
            )
        ]
        return ft.Column(rows, spacing=7)

    def show_message(message: str, error=False, success=False):
        status_text.value = message
        status_text.color = DANGER if error else (SUCCESS if success else MUTED)
        page.show_dialog(
            ft.SnackBar(
                content=ft.Text(
                    message,
                    color=ft.Colors.BLACK,
                )
            )
        )
        page.update()

    def read_template_path(name):
        path = template_paths.get(name)
        if path is None or not path.exists():
            filename = next((filename for title, filename in TEMPLATE_FILES if title == name), name)
            raise FileNotFoundError(f"فایل {filename} پیدا نشد.")
        try:
            with Image.open(path) as test_image:
                test_image.verify()
            with Image.open(path) as checked:
                if checked.size != TEMPLATE_SIZE:
                    raise ValueError(
                        f"اندازهٔ زمینه باید دقیقاً {TEMPLATE_SIZE[0]}×{TEMPLATE_SIZE[1]} باشد؛ "
                        f"اندازهٔ فعلی {checked.width}×{checked.height} است."
                    )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"فایل زمینه معتبر نیست: {path.name}") from exc
        return path

    def get_session_data():
        return {
            "number": txt_number.value or "",
            "title": txt_title.value or "",
            "subject": txt_subject.value or "",
            "presenter": txt_presenter.value or "",
            "date": txt_date.value or "",
        }

    def build_share_text():
        data = get_session_data()
        return (
            f"{data['number']} {data['title']}\n\n"
            f"با موضوع\n{data['subject']}\n\n"
            f"در محضر\n{data['presenter']}\n\n"
            f"{data['date']}\n\n"
            "https://eitaa.com/shfeghhi\n"
            "https://t.me/shfeghhi"
        )

    def save_draft(include_photo=False):
        photo_saved = bool(draft_state.get("photo_saved", False))
        if include_photo:
            photo_saved = False
            if selected_photo_bytes["value"]:
                try:
                    draft_photo_path.write_bytes(selected_photo_bytes["value"])
                    photo_saved = True
                except OSError:
                    photo_saved = False

        data = {
            "template": selected_template["name"],
            "number": txt_number.value or "",
            "title": txt_title.value or "",
            "subject": txt_subject.value or "",
            "presenter": txt_presenter.value or "",
            "date": txt_date.value or "",
            "photo_saved": photo_saved,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            atomic_write_json(draft_path, data)
            draft_state.clear()
            draft_state.update(data)
        except OSError:
            pass

    def save_draft_from_event():
        # هنگام تایپ، ذخیرهٔ دیسک را debounce می‌کنیم تا I/O روی هر کلید
        # باعث کندی رابط کاربری نشود.
        previous = draft_save_task.get("value")
        if previous is not None and not previous.done():
            previous.cancel()

        async def delayed_save():
            try:
                await asyncio.sleep(0.35)
                save_draft(include_photo=False)
            except asyncio.CancelledError:
                return

        draft_save_task["value"] = page.run_task(delayed_save)

    def draft_available():
        return bool(draft_state.get("template")) and draft_path.exists()

    async def restore_draft(e=None):
        state = read_json(draft_path, {})
        if not isinstance(state, dict) or not state.get("template"):
            show_message("پیش‌نویس قابل بازیابی پیدا نشد.", error=True)
            return

        template_name = state.get("template")
        if template_name not in template_paths:
            show_message("زمینهٔ ذخیره‌شده دیگر در پوشه assets وجود ندارد.", error=True)
            return

        # برای حفظ روند اجباری انتخاب زمینه، اطلاعات پیش‌نویس تا زمان
        # لمس همان زمینه نگه داشته می‌شود و مستقیماً وارد فرم نمی‌شویم.
        pending_restore["state"] = state
        show_template_selection()
        status_text.value = (
            f"پیش‌نویس آماده است؛ ابتدا «{template_name}» را از میان هفت زمینه انتخاب کنید."
        )
        status_text.color = GOLD
        page.update()

    # ========================================================
    # پیش‌نمایش پوستر
    # ========================================================
    preview_image = ft.Image(
        src="",
        width=290,
        height=435,
        fit=ft.BoxFit.CONTAIN,
        visible=False,
    )

    preview_placeholder = ft.Container(
        width=290,
        height=435,
        bgcolor="#0C1F1A",
        border=ft.Border.all(1, "#2A433C"),
        border_radius=18,
        alignment=ft.Alignment.CENTER,
        content=ft.Column(
            [
                ft.Icon(ft.Icons.IMAGE_OUTLINED, size=46, color=MUTED),
                ft.Text("پیش‌نمایش زنده", color=MUTED, size=15),
                ft.Text(
                    "متن‌ها و عکس شما همین‌جا\nنمایش داده می‌شوند.",
                    color="#6E8179",
                    size=11,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
        ),
    )

    preview_stack = ft.Stack(
        [
            preview_placeholder,
            ft.Container(
                content=preview_image,
                alignment=ft.Alignment.CENTER,
            ),
        ],
        width=300,
        height=445,
    )

    def render_poster_bytes(template_path_override=None, data_override=None, photo_bytes_override=None):
        active_template_path = template_path_override or selected_template["path"]
        if not active_template_path:
            return None
        if not active_template_path.exists():
            return None

        data = data_override if data_override is not None else get_session_data()
        photo_bytes = (
            photo_bytes_override
            if photo_bytes_override is not None
            else selected_photo_bytes["value"]
        )
        image = Image.open(active_template_path).convert("RGBA")
        draw = ImageDraw.Draw(image)

        font_number = load_font(str(font_path), 43)
        font_title = load_font(str(font_path), 50)
        font_subject = load_font(str(font_path), 55)
        font_presenter = load_font(str(font_path), 26)
        font_date = load_font(str(font_path), 25)

        draw_centered_text(draw, data["number"], BOX_TOP_NUMBER, font_number, GOLD)
        draw_centered_text(draw, data["title"], BOX_TOP_TITLE, font_title, GOLD)

        lines = data["subject"].splitlines() or [""]
        draw_fitted_centered_lines(
            draw,
            lines,
            BOX_MIDDLE,
            str(font_path),
            base_size=SUBJECT_BASE_FONT_SIZE,
            base_line_spacing=SUBJECT_BASE_LINE_SPACING,
            fill_color=NAVY,
        )

        if photo_bytes:
            photo = open_normalized_image(photo_bytes)
            target_w, target_h = BOX_PHOTO["w"], BOX_PHOTO["h"]
            photo.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            canvas.alpha_composite(
                photo,
                ((target_w - photo.width) // 2, (target_h - photo.height) // 2),
            )
            image.alpha_composite(
                canvas,
                (
                    int(BOX_PHOTO["x"] - target_w / 2),
                    int(BOX_PHOTO["y"] - target_h / 2),
                ),
            )

        presenter_lines = data["presenter"].splitlines() or [""]
        draw_fitted_centered_lines(
            draw,
            presenter_lines,
            BOX_PRESENTER,
            str(font_path),
            base_size=PRESENTER_BASE_FONT_SIZE,
            base_line_spacing=PRESENTER_BASE_LINE_SPACING,
            fill_color=NAVY,
        )

        draw_centered_text(draw, data["date"], BOX_DATE, font_date, NAVY)

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=95)
        return buffer.getvalue()

    def render_live_preview(template_path_override=None, data_override=None, photo_bytes_override=None, update_current=True):
        try:
            data = render_poster_bytes(
                template_path_override=template_path_override,
                data_override=data_override,
                photo_bytes_override=photo_bytes_override,
            )
            if not data:
                preview_image.visible = False
                preview_placeholder.visible = True
                return
            preview_image.src = data
            preview_image.visible = True
            preview_placeholder.visible = False
            if update_current:
                current_poster["bytes"] = data
                current_poster["path"] = None
        except Exception as exc:
            show_message(f"خطا در پیش‌نمایش زنده: {exc}", error=True)

    def clear_preview():
        preview_image.visible = False
        preview_image.src = ""
        preview_placeholder.visible = True
        current_poster["path"] = None
        current_poster["bytes"] = None

    # ========================================================
    # کراپ سریع ۳:۴
    # ========================================================
    CROP_WIDTH = 288
    CROP_HEIGHT = 384
    CROP_RATIO = 3 / 4
    CROP_MIN_ZOOM = 1.0
    CROP_MAX_ZOOM = 4.0
    CROP_DISPLAY_MAX = 1400

    crop_state = {
        "original_bytes": None,
        "source_image": None,
        "display_image": None,
        "display_factor": 1.0,
        "file_name": "",
        "center_x": 0.0,
        "center_y": 0.0,
        "zoom": 1.0,
        "scale_active": False,
        "scale_start_zoom": 1.0,
    }

    def clamp(value, low, high):
        return max(low, min(high, value))

    def source_crop_geometry():
        source = crop_state["source_image"]
        if source is None:
            return 0.0, 0.0
        source_ratio = source.width / source.height
        if source_ratio >= CROP_RATIO:
            base_h = float(source.height)
            base_w = base_h * CROP_RATIO
        else:
            base_w = float(source.width)
            base_h = base_w / CROP_RATIO
        zoom = clamp(
            float(crop_state["zoom"]),
            CROP_MIN_ZOOM,
            CROP_MAX_ZOOM,
        )
        return base_w / zoom, base_h / zoom

    def clamp_crop_center(crop_w=None, crop_h=None):
        source = crop_state["source_image"]
        if source is None:
            return
        if crop_w is None or crop_h is None:
            crop_w, crop_h = source_crop_geometry()

        half_w = crop_w / 2.0
        half_h = crop_h / 2.0
        min_x = half_w
        max_x = max(half_w, source.width - half_w)
        min_y = half_h
        max_y = max(half_h, source.height - half_h)

        crop_state["center_x"] = clamp(float(crop_state["center_x"]), min_x, max_x)
        crop_state["center_y"] = clamp(float(crop_state["center_y"]), min_y, max_y)

    def get_crop_box():
        source = crop_state["source_image"]
        if source is None:
            raise ValueError("تصویری برای برش وجود ندارد.")

        crop_w, crop_h = source_crop_geometry()
        clamp_crop_center(crop_w, crop_h)

        left = int(round(crop_state["center_x"] - crop_w / 2.0))
        top = int(round(crop_state["center_y"] - crop_h / 2.0))
        right = int(round(crop_state["center_x"] + crop_w / 2.0))
        bottom = int(round(crop_state["center_y"] + crop_h / 2.0))

        left = int(clamp(left, 0, max(0, source.width - 2)))
        top = int(clamp(top, 0, max(0, source.height - 2)))
        right = int(clamp(right, left + 1, source.width))
        bottom = int(clamp(bottom, top + 1, source.height))
        return left, top, right, bottom

    def build_crop_display_image(source: Image.Image):
        display = source.copy()
        display.thumbnail(
            (CROP_DISPLAY_MAX, CROP_DISPLAY_MAX),
            Image.Resampling.LANCZOS,
        )
        output = io.BytesIO()
        # هنگام نمایش، PNG شفاف باقی می‌ماند و فقط یک بار تولید می‌شود.
        display.save(output, format="PNG", optimize=True)
        factor = display.width / source.width if source.width else 1.0
        return display, factor, output.getvalue()

    crop_fast_image = ft.Image(
        src="",
        width=CROP_WIDTH,
        height=CROP_HEIGHT,
        fit=ft.BoxFit.FILL,
        anti_alias=True,
        visible=True,
    )

    crop_border = ft.Container(
        width=CROP_WIDTH,
        height=CROP_HEIGHT,
        bgcolor="#00000000",
        border=ft.Border.all(3, GOLD),
        border_radius=12,
    )

    crop_ratio_badge = ft.Container(
        padding=ft.Padding.symmetric(horizontal=9, vertical=5),
        bgcolor=BLACK_GLASS,
        border_radius=12,
        content=ft.Text(
            "خروجی ۳ : ۴",
            color=CREAM,
            size=11,
            weight=ft.FontWeight.BOLD,
        ),
    )

    crop_stage_stack = ft.Stack(
        controls=[
            crop_fast_image,
            crop_border,
            ft.Container(
                alignment=ft.Alignment.TOP_RIGHT,
                padding=10,
                content=crop_ratio_badge,
            ),
        ],
        width=CROP_WIDTH,
        height=CROP_HEIGHT,
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )

    crop_stage = ft.Container(
        width=CROP_WIDTH,
        height=CROP_HEIGHT,
        bgcolor="#181F1C",
        border_radius=12,
        alignment=ft.Alignment.CENTER,
        content=crop_stage_stack,
    )

    crop_gesture = ft.GestureDetector(
        drag_interval=16,
        content=crop_stage,
    )

    crop_zoom_value = ft.Text(
        "1.0×",
        color=GOLD,
        size=13,
        weight=ft.FontWeight.BOLD,
        text_align=ft.TextAlign.CENTER,
    )
    crop_position_text = ft.Text(
        "تصویر در مرکز کادر قرار گرفت",
        color=MUTED,
        size=10,
        text_align=ft.TextAlign.CENTER,
    )
    crop_status = ft.Text(
        "",
        color=MUTED,
        size=10,
        text_align=ft.TextAlign.CENTER,
    )

    def update_crop_visual(update_slider=False):
        source = crop_state["source_image"]
        display = crop_state["display_image"]
        if source is None or display is None:
            return

        crop_w, crop_h = source_crop_geometry()
        clamp_crop_center(crop_w, crop_h)

        preview_factor = crop_state["display_factor"]
        ui_scale = CROP_HEIGHT / (crop_h * preview_factor)
        display_w = display.width * ui_scale
        display_h = display.height * ui_scale

        crop_fast_image.width = display_w
        crop_fast_image.height = display_h
        crop_fast_image.left = CROP_WIDTH / 2 - crop_state["center_x"] * preview_factor * ui_scale
        crop_fast_image.top = CROP_HEIGHT / 2 - crop_state["center_y"] * preview_factor * ui_scale

        crop_zoom_value.value = f"{crop_state['zoom']:.1f}×"
        if update_slider:
            crop_zoom_slider.value = float(crop_state["zoom"])

        # در Flet 1.0 پایان event به‌صورت خودکار update می‌شود؛
        # بنابراین در حلقهٔ پرسرعت کراپ هیچ update جداگانه‌ای ارسال نمی‌کنیم.

    def zoom_crop_at(new_zoom, focal_x=None, focal_y=None):
        source = crop_state["source_image"]
        if source is None:
            return

        old_crop_w, old_crop_h = source_crop_geometry()
        old_center_x = float(crop_state["center_x"])
        old_center_y = float(crop_state["center_y"])

        if focal_x is None:
            focal_x = CROP_WIDTH / 2
        if focal_y is None:
            focal_y = CROP_HEIGHT / 2

        focal_x = clamp(float(focal_x), 0.0, float(CROP_WIDTH))
        focal_y = clamp(float(focal_y), 0.0, float(CROP_HEIGHT))

        source_point_x = (
            old_center_x
            - old_crop_w / 2
            + (focal_x / CROP_WIDTH) * old_crop_w
        )
        source_point_y = (
            old_center_y
            - old_crop_h / 2
            + (focal_y / CROP_HEIGHT) * old_crop_h
        )

        crop_state["zoom"] = clamp(float(new_zoom), CROP_MIN_ZOOM, CROP_MAX_ZOOM)
        new_crop_w, new_crop_h = source_crop_geometry()

        crop_state["center_x"] = source_point_x - (
            focal_x / CROP_WIDTH - 0.5
        ) * new_crop_w
        crop_state["center_y"] = source_point_y - (
            focal_y / CROP_HEIGHT - 0.5
        ) * new_crop_h
        clamp_crop_center(new_crop_w, new_crop_h)
        update_crop_visual(update_slider=True)

    def on_crop_zoom_change(e):
        zoom_crop_at(float(crop_zoom_slider.value or 1.0))

    def change_crop_zoom(amount):
        zoom_crop_at(float(crop_state["zoom"]) + amount)

    def reset_crop_view(e=None):
        source = crop_state["source_image"]
        if source is None:
            return
        crop_state["zoom"] = 1.0
        crop_state["center_x"] = source.width / 2
        crop_state["center_y"] = source.height / 2
        crop_state["scale_start_zoom"] = 1.0
        clamp_crop_center()
        update_crop_visual(update_slider=True)
        crop_position_text.value = "کادر به حالت اولیه بازگشت"
        crop_position_text.update()

    crop_zoom_slider = ft.Slider(
        min=CROP_MIN_ZOOM,
        max=CROP_MAX_ZOOM,
        divisions=30,
        value=1.0,
        label="{value}×",
        active_color=GOLD,
        inactive_color=BORDER,
        on_change=on_crop_zoom_change,
    )

    btn_crop_minus = ft.Button(
        content="−",
        icon=ft.Icons.ZOOM_OUT,
        on_click=lambda e: change_crop_zoom(-0.1),
        style=ft.ButtonStyle(bgcolor=PANEL_2, color=CREAM),
    )
    btn_crop_plus = ft.Button(
        content="+",
        icon=ft.Icons.ZOOM_IN,
        on_click=lambda e: change_crop_zoom(0.1),
        style=ft.ButtonStyle(bgcolor=GOLD, color=BG),
    )

    crop_zoom_controls = card_container(
        ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("بزرگنمایی", size=12, color=CREAM),
                        ft.Container(expand=True),
                        crop_zoom_value,
                    ]
                ),
                ft.Row(
                    [
                        btn_crop_minus,
                        ft.Container(expand=True, content=crop_zoom_slider),
                        btn_crop_plus,
                    ],
                    spacing=6,
                ),
                ft.Row(
                    [
                        ft.Text("۱×", size=10, color=MUTED),
                        ft.Container(expand=True),
                        ft.Text("۴×", size=10, color=MUTED),
                    ]
                ),
            ],
            spacing=2,
        ),
        padding=10,
        radius=14,
    )

    btn_crop_reset = ft.Button(
        content="بازنشانی کادر",
        icon=ft.Icons.RESTART_ALT,
        on_click=reset_crop_view,
        style=ft.ButtonStyle(bgcolor=PANEL_2, color=CREAM),
    )

    crop_tip = card_container(
        ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(ft.Icons.TOUCH_APP, size=18, color=GOLD),
                        ft.Text("یک انگشت: جابه‌جایی عکس", size=11, color=CREAM),
                    ],
                    spacing=7,
                ),
                ft.Row(
                    [
                        ft.Icon(ft.Icons.ZOOM_IN, size=18, color=GOLD),
                        ft.Text("دو انگشت: زوم و حرکت هم‌زمان", size=11, color=CREAM),
                    ],
                    spacing=7,
                ),
                ft.Text(
                    "در هنگام حرکت، فقط هندسهٔ تصویر عوض می‌شود؛ بنابراین پاسخ‌گویی سریع‌تر از نسخه قبلی است.",
                    size=10,
                    color=MUTED,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
            spacing=6,
        ),
        padding=11,
        radius=14,
    )

    btn_remove_background = ft.Button(
        content=(
            "حذف هوشمند زمینه" if BACKGROUND_REMOVAL_AVAILABLE
            else "حذف هوشمند زمینه — موتور در دسترس نیست"
        ),
        icon=ft.Icons.AUTO_FIX_HIGH,
        disabled=not BACKGROUND_REMOVAL_AVAILABLE,
    )

    def crop_pan_start(e: ft.DragStartEvent):
        crop_state["scale_active"] = False

    def crop_pan_update(e: ft.DragUpdateEvent):
        if crop_state["source_image"] is None or crop_state["scale_active"]:
            return
        delta = getattr(e, "local_delta", None)
        if delta is None:
            return
        crop_w, crop_h = source_crop_geometry()
        dx = float(getattr(delta, "x", 0.0) or 0.0)
        dy = float(getattr(delta, "y", 0.0) or 0.0)

        crop_state["center_x"] -= (dx / CROP_WIDTH) * crop_w
        crop_state["center_y"] -= (dy / CROP_HEIGHT) * crop_h
        clamp_crop_center(crop_w, crop_h)
        update_crop_visual(update_slider=False)

    def crop_pan_end(e: ft.DragEndEvent):
        crop_state["scale_active"] = False

    def crop_scale_start(e: ft.ScaleStartEvent):
        pointer_count = int(getattr(e, "pointer_count", 1) or 1)
        crop_state["scale_active"] = pointer_count >= 2
        crop_state["scale_start_zoom"] = float(crop_state["zoom"])

    def crop_scale_update(e: ft.ScaleUpdateEvent):
        if crop_state["source_image"] is None:
            return
        scale = float(getattr(e, "scale", 1.0) or 1.0)
        requested_zoom = crop_state["scale_start_zoom"] * scale

        focal = getattr(e, "local_focal_point", None)
        if focal is None:
            focal = getattr(e, "focal_point", None)
        if focal is None:
            focal_x = CROP_WIDTH / 2
            focal_y = CROP_HEIGHT / 2
        else:
            focal_x = float(getattr(focal, "x", CROP_WIDTH / 2))
            focal_y = float(getattr(focal, "y", CROP_HEIGHT / 2))

        zoom_crop_at(requested_zoom, focal_x=focal_x, focal_y=focal_y)

        focal_delta = getattr(e, "focal_point_delta", None)
        if focal_delta is not None:
            crop_w, crop_h = source_crop_geometry()
            dx = float(getattr(focal_delta, "x", 0.0) or 0.0)
            dy = float(getattr(focal_delta, "y", 0.0) or 0.0)
            crop_state["center_x"] -= (dx / CROP_WIDTH) * crop_w
            crop_state["center_y"] -= (dy / CROP_HEIGHT) * crop_h
            clamp_crop_center(crop_w, crop_h)
            update_crop_visual(update_slider=False)

    def crop_scale_end(e: ft.ScaleEndEvent):
        crop_state["scale_active"] = False

    crop_gesture.on_pan_start = crop_pan_start
    crop_gesture.on_pan_update = crop_pan_update
    crop_gesture.on_pan_end = crop_pan_end
    crop_gesture.on_scale_start = crop_scale_start
    crop_gesture.on_scale_update = crop_scale_update
    crop_gesture.on_scale_end = crop_scale_end

    def show_crop_editor():
        # کراپ به‌صورت یک صفحهٔ واقعی در همان برنامه نمایش داده می‌شود؛
        # نوار پیمایش پایین در این صفحه مخفی است تا فضای بیشتری برای عکس بماند.
        page.navigation_bar.visible = False
        page_content.content = crop_editor_view
        page.update()

    def prepare_crop(raw_bytes: bytes, file_name: str):
        try:
            source = open_normalized_image(raw_bytes)
            if source.width < 3 or source.height < 4:
                raise ValueError("ابعاد عکس برای برش ۳:۴ کافی نیست.")

            display, factor, display_bytes = build_crop_display_image(source)
            crop_state["original_bytes"] = raw_bytes
            crop_state["source_image"] = source
            crop_state["display_image"] = display
            crop_state["display_factor"] = factor
            crop_state["file_name"] = file_name
            crop_state["zoom"] = 1.0
            crop_state["center_x"] = source.width / 2
            crop_state["center_y"] = source.height / 2
            crop_state["scale_active"] = False
            crop_state["scale_start_zoom"] = 1.0

            crop_fast_image.src = display_bytes
            clamp_crop_center()
            update_crop_visual(update_slider=True)
            crop_status.value = f"{file_name} • عکس هنوز روی پوستر اعمال نشده است"
            crop_status.color = MUTED
            crop_position_text.value = "تصویر در مرکز کادر قرار گرفت"
            crop_position_text.color = MUTED
            show_crop_editor()
        except Exception as exc:
            show_message(f"خطا در آماده‌سازی عکس برای برش: {exc}", error=True)

    def get_cropped_photo_bytes() -> bytes:
        source = crop_state["source_image"]
        if source is None:
            raise ValueError("تصویری برای کراپ وجود ندارد.")
        left, top, right, bottom = get_crop_box()
        cropped = source.crop((left, top, right, bottom)).convert("RGBA")
        cropped = ImageOps.fit(
            cropped,
            (600, 800),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        output = io.BytesIO()
        cropped.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def accept_crop(e=None):
        try:
            cropped_bytes = get_cropped_photo_bytes()
            selected_photo_bytes["value"] = cropped_bytes
            file_name = crop_state["file_name"] or "عکس"
            txt_photo_status.value = f"{file_name} • برش ۳:۴ اعمال شد"
            txt_photo_status.color = SUCCESS
            status_text.value = "کراپ تأیید شد؛ عکس جدید روی پیش‌نمایش اعمال شد."
            status_text.color = SUCCESS
            save_draft(include_photo=True)
            show_main_editor()
            render_live_preview()
            page.update()
        except Exception as exc:
            crop_status.value = f"خطا در اعمال کراپ: {exc}"
            crop_status.color = DANGER
            page.update()

    def cancel_crop(e=None):
        # عکس قبلی عمداً دست‌نخورده باقی می‌ماند.
        crop_state["original_bytes"] = None
        crop_state["source_image"] = None
        crop_state["display_image"] = None
        crop_state["file_name"] = ""
        crop_state["scale_active"] = False
        show_main_editor()
        status_text.value = "برش لغو شد؛ عکس قبلی بدون تغییر باقی ماند."
        status_text.color = MUTED
        page.update()

    btn_crop_cancel = ft.Button(
        content="انصراف",
        icon=ft.Icons.CLOSE,
        on_click=cancel_crop,
        style=ft.ButtonStyle(bgcolor=PANEL_2, color=CREAM),
    )
    btn_crop_apply = ft.Button(
        content="تأیید و اعمال",
        icon=ft.Icons.CHECK,
        on_click=accept_crop,
        style=ft.ButtonStyle(bgcolor=GOLD, color=BG),
    )

    # ========================================================
    # حذف زمینه هوشمند — دانلود خودکار مدل + ONNX Runtime + U²-NetP
    # ========================================================
    def verify_background_model(path: Path) -> bool:
        try:
            if not path.exists() or path.stat().st_size < 100000:
                return False
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            return digest.lower() == BACKGROUND_MODEL_SHA256.lower()
        except Exception:
            return False

    def download_background_model_sync(target_path: Path):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(target_path.suffix + ".part")

        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

        request = Request(
            BACKGROUND_MODEL_URL,
            headers={
                "User-Agent": "PousterCreator/2.0",
                "Accept": "application/octet-stream",
            },
        )

        try:
            with urlopen(request, timeout=90) as response, open(temp_path, "wb") as output:
                while True:
                    chunk = response.read(BACKGROUND_MODEL_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    output.write(chunk)

            if not verify_background_model(temp_path):
                raise RuntimeError("دانلود مدل کامل یا معتبر نبود.")

            temp_path.replace(target_path)
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            raise

    async def ensure_background_model():
        if not BACKGROUND_REMOVAL_AVAILABLE:
            raise RuntimeError(
                "موتور حذف زمینه در این نسخه از برنامه در دسترس نیست. "
                "onnxruntime در APK/محیط فعلی قرار نگرفته است."
            )

        if verify_background_model(bg_model_path):
            return

        # فایل خراب یا ناقص قبلی حذف می‌شود و نسخهٔ معتبر دوباره دریافت می‌شود.
        try:
            if bg_model_path.exists():
                bg_model_path.unlink()
        except OSError:
            pass

        crop_status.value = "مدل حذف زمینه هنوز روی دستگاه نیست؛ در حال دریافت خودکار…"
        crop_status.color = GOLD
        page.update()

        await asyncio.to_thread(download_background_model_sync, bg_model_path)
        crop_status.value = "مدل با موفقیت دریافت شد؛ در حال آماده‌سازی موتور…"
        crop_status.color = MUTED
        page.update()

    def _build_bg_session(optimization_level=None):
        if not BACKGROUND_REMOVAL_AVAILABLE:
            raise RuntimeError(
                "موتور حذف زمینه در این نسخه از برنامه در دسترس نیست. "
                "وابستگی onnxruntime در APK/محیط فعلی موجود نیست."
            )
        if not bg_model_path.exists():
            raise FileNotFoundError("مدل حذف زمینه هنوز دانلود نشده است.")

        digest = hashlib.sha256(bg_model_path.read_bytes()).hexdigest()
        if digest.lower() != BACKGROUND_MODEL_SHA256.lower():
            raise RuntimeError(
                "فایل مدل حذف زمینه ناقص یا خراب است. برنامه دوباره آن را دریافت می‌کند."
            )

        options = ort.SessionOptions()
        # Android: از layout optimization (NCHWc/ReorderInput) صرف‌نظر می‌کنیم.
        # U²-NetP ورودی ثابت 1×3×320×320 دارد و BASIC برای این مسیر کافی است.
        if optimization_level is None:
            optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        try:
            options.graph_optimization_level = optimization_level
        except Exception:
            pass
        try:
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        except Exception:
            pass
        try:
            options.enable_cpu_mem_arena = False
        except Exception:
            pass
        try:
            options.enable_mem_pattern = False
        except Exception:
            pass
        try:
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
        except Exception:
            pass

        session = ort.InferenceSession(
            str(bg_model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        # قرارداد ورودی را قبل از inference کنترل می‌کنیم تا مدل اشتباه
        # یا ورودی غیرمنتظره هرگز باعث allocation عظیم نشود.
        input_meta = session.get_inputs()[0]
        expected_shape = [1, 3, 320, 320]
        if list(input_meta.shape) != expected_shape:
            raise RuntimeError(
                f"مدل حذف زمینه ورودی غیرمنتظره دارد: {input_meta.shape}; "
                f"باید {expected_shape} باشد."
            )
        if str(input_meta.type).lower() != "tensor(float)":
            raise RuntimeError(
                f"نوع ورودی مدل پشتیبانی نمی‌شود: {input_meta.type}"
            )
        return session

    def get_bg_session():
        if bg_session["value"] is None:
            bg_session["value"] = _build_bg_session()
        return bg_session["value"]

    def preprocess_u2netp(image: Image.Image):
        rgb = image.convert("RGB")
        resized = rgb.resize(SEGMENT_INPUT_SIZE, Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return arr

    def postprocess_u2netp(image_size, prediction):
        pred = prediction[0, 0]
        pred_min = float(pred.min())
        pred_max = float(pred.max())
        if pred_max - pred_min > 1e-8:
            pred = (pred - pred_min) / (pred_max - pred_min)
        else:
            pred = np.zeros_like(pred, dtype=np.float32)

        mask = Image.fromarray(
            np.clip(pred * 255.0, 0, 255).astype(np.uint8)
        )
        mask = mask.resize(image_size, Image.Resampling.LANCZOS)

        # لبه‌های نیمه‌شفاف برای مو/عمامه/لباس بهتر حفظ می‌شوند.
        mask = mask.filter(ImageFilter.GaussianBlur(0.35))
        mask_np = np.asarray(mask, dtype=np.uint8)
        alpha_np = mask_np.astype(np.uint16)
        alpha_np = np.where(alpha_np < 18, 0, alpha_np)
        alpha_np = np.where(alpha_np > 242, 255, alpha_np)
        alpha = Image.fromarray(alpha_np.astype(np.uint8))
        return alpha

    def remove_background_sync(raw_bytes: bytes) -> bytes:
        session = get_bg_session()
        image = open_normalized_image(raw_bytes)

        # برای موبایل‌های معمولی، مدل روی تصویری تا ۱۸۰۰ پیکسل کار می‌کند؛
        # این کار مصرف RAM و زمان را کاهش می‌دهد و برای محل عکس پوستر بیش از کافی است.
        if max(image.width, image.height) > SEGMENT_MAX_SIDE:
            scale = SEGMENT_MAX_SIDE / max(image.width, image.height)
            work_size = (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            )
            work_image = image.resize(work_size, Image.Resampling.LANCZOS)
        else:
            work_image = image

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        tensor = preprocess_u2netp(work_image)

        try:
            outputs = session.run([output_name], {input_name: tensor})
        except Exception as first_exc:
            # در بعضی نسخه‌های Android/ORT، اگر allocation غیرعادی رخ دهد،
            # یک session کاملاً بدون graph optimization امتحان می‌کنیم.
            if "bad allocation" not in str(first_exc).lower():
                raise
            bg_session["value"] = None
            gc.collect()
            safe_session = _build_bg_session(
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            bg_session["value"] = safe_session
            output_name = safe_session.get_outputs()[0].name
            input_name = safe_session.get_inputs()[0].name
            outputs = safe_session.run([output_name], {input_name: tensor})

        if not outputs:
            raise RuntimeError("مدل حذف زمینه خروجی معتبری برنگرداند.")

        prediction = outputs[0]
        del tensor, outputs
        alpha = postprocess_u2netp(work_image.size, prediction)
        del prediction
        rgba = work_image.convert("RGBA")
        rgba.putalpha(alpha)

        output = io.BytesIO()
        rgba.save(output, format="PNG", optimize=True)
        return output.getvalue()

    async def remove_background_async(e=None):
        if bg_removing["value"] or crop_state["source_image"] is None:
            return
        if not BACKGROUND_REMOVAL_AVAILABLE:
            show_message(
                "موتور حذف زمینه در این محیط در دسترس نیست.",
                error=True,
            )
            return

        bg_removing["value"] = True
        btn_remove_background.disabled = True
        crop_status.value = "در حال حذف هوشمند زمینه…"
        crop_status.color = MUTED
        btn_remove_background.update()
        crop_status.update()
        try:
            await ensure_background_model()
            raw = crop_state["original_bytes"] or b""
            cleaned = await asyncio.to_thread(remove_background_sync, raw)
            source = open_normalized_image(cleaned)
            display, factor, display_bytes = build_crop_display_image(source)

            crop_state["source_image"] = source
            crop_state["original_bytes"] = cleaned
            crop_state["display_image"] = display
            crop_state["display_factor"] = factor
            crop_state["zoom"] = 1.0
            crop_state["center_x"] = source.width / 2
            crop_state["center_y"] = source.height / 2
            crop_state["scale_active"] = False
            crop_state["scale_start_zoom"] = 1.0

            crop_fast_image.src = display_bytes
            clamp_crop_center()
            update_crop_visual(update_slider=True)
            crop_status.value = "زمینه حذف شد؛ اکنون کادر ۳:۴ را تنظیم و تأیید کنید."
            crop_status.color = SUCCESS
            show_message("زمینه با مدل سبک محلی حذف شد.", success=True)
        except Exception as exc:
            crop_status.value = f"حذف زمینه انجام نشد: {exc}"
            crop_status.color = DANGER
            show_message(f"حذف زمینه انجام نشد: {exc}", error=True)
        finally:
            bg_removing["value"] = False
            btn_remove_background.disabled = False
            btn_remove_background.update()
            crop_status.update()

    btn_remove_background.on_click = remove_background_async

    crop_editor_header = ft.Container(
        padding=ft.Padding.symmetric(horizontal=14, vertical=13),
        bgcolor=PANEL,
        border=ft.Border.only(bottom=ft.BorderSide(width=1, color="#1C3430")),
        content=ft.Row(
            [
                ft.Button(
                    content="برگشت",
                    icon=ft.Icons.ARROW_BACK,
                    on_click=lambda e: show_main_editor(),
                    style=ft.ButtonStyle(bgcolor=PANEL_2, color=CREAM),
                ),
                ft.Column(
                    [
                        ft.Text(
                            "تنظیم و برش حرفه‌ای عکس",
                            size=19,
                            weight=ft.FontWeight.BOLD,
                            color=CREAM,
                        ),
                        ft.Text(
                            "کادر دقیق ۳:۴ • حرکت روان • زوم لمسی",
                            size=10,
                            color=MUTED,
                        ),
                    ],
                    spacing=1,
                    expand=True,
                ),
                ft.Icon(ft.Icons.CROP, color=GOLD, size=24),
            ],
            spacing=10,
        ),
    )

    crop_actions = ft.Row(
        [
            ft.Container(expand=True, height=58, content=btn_crop_cancel),
            ft.Container(expand=True, height=58, content=btn_crop_apply),
        ],
        spacing=10,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    crop_editor_body = ft.Container(
        expand=True,
        padding=ft.Padding.symmetric(horizontal=12, vertical=14),
        content=ft.Column(
            [
                ft.Container(
                    padding=10,
                    bgcolor=PANEL,
                    border_radius=16,
                    border=ft.Border.all(1, BORDER),
                    alignment=ft.Alignment.CENTER,
                    content=crop_gesture,
                ),
                ft.Row(
                    [btn_remove_background, btn_crop_reset],
                    spacing=8,
                    alignment=ft.MainAxisAlignment.CENTER,
                    wrap=True,
                ),
                crop_tip,
                crop_zoom_controls,
                crop_position_text,
                crop_status,
                ft.Container(height=8),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        ),
    )

    # نوار اعمال بیرون از ScrollView و داخل SafeArea است تا در Android
    # پشت نوار ناوبری/gesture system پنهان نشود.
    crop_editor_footer = ft.SafeArea(
        maintain_bottom_view_padding=True,
        content=ft.Container(
            height=82,
            padding=ft.Padding.symmetric(horizontal=12, vertical=9),
            bgcolor=PANEL,
            border=ft.Border.only(top=ft.BorderSide(width=1, color=BORDER)),
            content=crop_actions,
        ),
    )

    crop_editor_view = ft.Container(
        expand=True,
        bgcolor=BG,
        content=ft.Column(
            [
                crop_editor_header,
                crop_editor_body,
                crop_editor_footer,
            ],
            spacing=0,
            expand=True,
        ),
    )

    # ========================================================
    # انتخاب عکس
    # ========================================================
    async def pick_photo(e=None):
        try:
            files = await file_picker.pick_files(
                allow_multiple=False,
                file_type=ft.FilePickerFileType.IMAGE,
                with_data=True,
            )
            if not files:
                return
            selected = files[0]
            raw_value = None

            try:
                if selected.bytes:
                    raw_value = normalize_image_bytes(selected.bytes)
            except Exception:
                raw_value = None

            if not raw_value and getattr(selected, "path", None):
                try:
                    raw_value = normalize_image_bytes(Path(selected.path).read_bytes())
                except Exception:
                    raw_value = None

            if not raw_value:
                show_message(
                    "فقط عکس ثابت JPG، PNG یا WEBP قابل استفاده است؛ GIF و ویدیو مجاز نیست.",
                    error=True,
                )
                return

            prepare_crop(raw_value, selected.name)
        except Exception as exc:
            show_message(f"انتخاب عکس انجام نشد: {exc}", error=True)

    btn_select_photo = ft.Button(
        content="انتخاب عکس از گالری",
        icon=ft.Icons.IMAGE_OUTLINED,
        on_click=pick_photo,
        style=ft.ButtonStyle(bgcolor=PANEL_2, color=CREAM),
    )

    # ========================================================
    # دکمه‌های فرم
    # ========================================================
    def set_today(e=None):
        txt_date.value = today_jalali_text()
        update_counts()
        render_live_preview()
        save_draft_from_event()
        status_text.value = "تاریخ امروز وارد شد."
        status_text.color = MUTED
        page.update()

    btn_today = ft.Button(
        content="تاریخ امروز",
        icon=ft.Icons.TODAY,
        on_click=set_today,
        style=ft.ButtonStyle(bgcolor=PANEL_2, color=CREAM),
    )

    def clear_form(e=None):
        # شمارهٔ جلسه همیشه از آخرین پوستر «موفقاً ساخته‌شده» + ۱ پیشنهاد می‌شود.
        # هیچ دکمه‌ای برای افزایش دستی شماره وجود ندارد.
        txt_number.value = current_next_session_text()
        txt_title.value = "جلسه شنبه های فقهی"
        txt_subject.value = ""
        txt_presenter.value = ""
        txt_date.value = today_jalali_text()
        selected_photo_bytes["value"] = None
        txt_photo_status.value = "عکسی انتخاب نشده — پس از انتخاب، وارد تنظیم عکس می‌شوید."
        txt_photo_status.color = MUTED
        crop_state["original_bytes"] = None
        crop_state["source_image"] = None
        crop_state["display_image"] = None
        update_counts()
        clear_preview()
        save_draft_from_event()
        status_text.value = "فرم به مقادیر پیشنهادی بازگردانی شد."
        status_text.color = MUTED
        page.update()

    btn_clear = ft.Button(
        content="بازنشانی فرم",
        icon=ft.Icons.CLEAR_ALL,
        on_click=clear_form,
        style=ft.ButtonStyle(bgcolor=PANEL_2, color=CREAM),
    )

    async def generate_and_save(e=None):
        if not selected_template["path"]:
            show_message("ابتدا یک زمینه انتخاب کنید.", error=True)
            show_template_selection()
            return
        if not any(
            [
                txt_number.value,
                txt_title.value,
                txt_subject.value,
                txt_presenter.value,
                txt_date.value,
            ]
        ):
            show_message("ابتدا اطلاعات جلسه را وارد کنید.", error=True)
            return
        if not selected_photo_bytes["value"]:
            show_message("ابتدا عکس ارائه‌دهنده را انتخاب و کراپ کنید.", error=True)
            return

        data = get_session_data()
        session_number = extract_session_number(data["number"])
        if session_number is None:
            session_number = current_next_session_number()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"poster_session_{session_number}_{timestamp}.jpg"
        output_path = output_dir / output_name

        try:
            create_poster(
                selected_template["path"],
                font_path,
                output_path,
                data,
                selected_photo_bytes["value"],
            )
            current_poster["bytes"] = output_path.read_bytes()
            current_poster["path"] = output_path
            preview_image.src = current_poster["bytes"]
            preview_image.visible = True
            preview_placeholder.visible = False

            if SESSION_MIN <= session_number <= SESSION_MAX:
                # فقط بعد از ساخت موفق ثبت می‌شود؛ بنابراین پوستر نیمه‌کاره
                # شماره جلسه را جلو نمی‌اندازد.
                app_state["last_session_number"] = session_number
            app_state["last_template"] = selected_template["name"]
            save_app_state()
            save_draft(include_photo=True)
            show_message(
                f"پوستر با موفقیت ساخته و ذخیره شد: {output_name}",
                success=True,
            )
        except Exception as exc:
            try:
                if output_path.exists():
                    output_path.unlink()
            except Exception:
                pass
            show_message(f"خطا در ساخت پوستر: {exc}", error=True)

    btn_generate = ft.Button(
        content="ساخت و ذخیره پوستر",
        icon=ft.Icons.SAVE,
        on_click=generate_and_save,
        style=ft.ButtonStyle(bgcolor=GOLD, color=BG),
    )

    async def save_current(e=None):
        data = current_poster["bytes"]
        if not data:
            show_message("ابتدا یک پوستر بسازید.", error=True)
            return
        try:
            path = await file_picker.save_file(
                dialog_title="ذخیره پوستر",
                file_name="poster.jpg",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["jpg"],
                src_bytes=data,
            )
            if page.web:
                show_message("دانلود پوستر آغاز شد.", success=True)
            elif path:
                show_message(f"پوستر ذخیره شد: {path}", success=True)
            else:
                show_message("عملیات ذخیره لغو شد.")
        except Exception as exc:
            show_message(f"خطا در ذخیره پوستر: {exc}", error=True)

    btn_save_as = ft.Button(
        content="ذخیره / دانلود",
        icon=ft.Icons.DOWNLOAD,
        on_click=save_current,
    )

    async def copy_session_text(e=None):
        try:
            await ft.Clipboard().set(build_share_text())
            show_message("متن جلسه در کلیپ‌بورد کپی شد.", success=True)
        except Exception as exc:
            show_message(f"کپی متن انجام نشد: {exc}", error=True)

    btn_copy_text = ft.Button(
        content="کپی متن جلسه",
        icon=ft.Icons.CONTENT_COPY,
        on_click=copy_session_text,
    )

    async def share_current(e=None):
        data = current_poster["bytes"]
        if not data:
            show_message("ابتدا یک پوستر بسازید.", error=True)
            return
        try:
            if page.web:
                await file_picker.save_file(
                    dialog_title="دانلود پوستر",
                    file_name="poster.jpg",
                    file_type=ft.FilePickerFileType.CUSTOM,
                    allowed_extensions=["jpg"],
                    src_bytes=data,
                )
                show_message("در نسخه وب، پوستر برای دانلود آماده شد.", success=True)
                return

            share_file = ft.ShareFile.from_bytes(
                data,
                mime_type="image/jpeg",
                name="poster.jpg",
            )
            result = await share_service.share_files(
                [share_file],
                title="اشتراک‌گذاری پوستر",
                text=build_share_text(),
            )
            show_message(f"وضعیت اشتراک‌گذاری: {result.status}", success=True)
        except Exception as exc:
            show_message(f"خطا در اشتراک‌گذاری: {exc}", error=True)

    btn_share = ft.Button(
        content="اشتراک‌گذاری",
        icon=ft.Icons.SHARE,
        on_click=share_current,
    )

    # ========================================================
    # شمارنده‌ها و پیش‌نمایش زنده
    # ========================================================
    def update_counts():
        subject_len = len((txt_subject.value or "").replace("\n", ""))
        presenter_len = len((txt_presenter.value or "").replace("\n", ""))
        subject_count.value = f"{to_persian_digits(subject_len)} نویسه"
        presenter_count.value = f"{to_persian_digits(presenter_len)} نویسه"

    def live_field_changed(e=None):
        update_counts()
        render_live_preview()
        save_draft_from_event()
        page.schedule_update()

    for control in (txt_number, txt_title, txt_subject, txt_presenter, txt_date):
        control.on_change = live_field_changed

    # ========================================================
    # انتخاب زمینه
    # ========================================================
    def select_template(template_name: str):
        try:
            path = read_template_path(template_name)
            if path.stat().st_size == 0:
                raise ValueError("فایل زمینه خالی است.")
        except Exception as exc:
            show_message(f"این زمینه قابل استفاده نیست: {exc}", error=True)
            return

        selected_template["name"] = template_name
        selected_template["path"] = template_paths[template_name]
        app_state["last_template"] = template_name
        save_app_state()

        restored = pending_restore.get("state")
        if isinstance(restored, dict) and restored.get("template") == template_name:
            txt_number.value = restored.get("number") or current_next_session_text()
            txt_title.value = restored.get("title") or "جلسه شنبه های فقهی"
            txt_subject.value = restored.get("subject") or ""
            txt_presenter.value = restored.get("presenter") or ""
            txt_date.value = restored.get("date") or today_jalali_text()
            if restored.get("photo_saved") and draft_photo_path.exists():
                try:
                    selected_photo_bytes["value"] = normalize_image_bytes(draft_photo_path.read_bytes())
                    txt_photo_status.value = "عکس پیش‌نویس نیز بازیابی شد."
                    txt_photo_status.color = SUCCESS
                except Exception:
                    selected_photo_bytes["value"] = None
                    txt_photo_status.value = "عکس پیش‌نویس قابل بازیابی نبود."
                    txt_photo_status.color = MUTED
            else:
                selected_photo_bytes["value"] = None
            pending_restore["state"] = None
            update_counts()
            render_live_preview()
            status_text.value = f"زمینه «{template_name}» انتخاب شد و پیش‌نویس بازیابی شد."
            status_text.color = SUCCESS
        else:
            # انتخاب یک زمینهٔ معمولی، حالت بازیابی معلق را لغو می‌کند.
            pending_restore["state"] = None
            selected_photo_bytes["value"] = None
            txt_photo_status.value = "عکسی انتخاب نشده — پس از انتخاب، وارد تنظیم عکس می‌شوید."
            txt_photo_status.color = MUTED
            status_text.value = f"زمینه «{template_name}» انتخاب شد. اکنون اطلاعات را وارد کنید."
            status_text.color = SUCCESS

        render_live_preview()
        show_main_editor()
        page.update()

    def template_card(template_name: str):
        path = template_paths[template_name]
        data = template_bytes.get(template_name)
        exists = path.exists() and bool(data)

        image_control = (
            ft.Image(
                src=data,
                width=132,
                height=198,
                fit=ft.BoxFit.CONTAIN,
                border_radius=12,
            )
            if exists
            else ft.Container(
                width=145,
                height=217.5,
                bgcolor="#0C1F1A",
                border_radius=12,
                alignment=ft.Alignment.CENTER,
                content=ft.Column(
                    [
                        ft.Icon(ft.Icons.IMAGE_OUTLINED, color=DANGER, size=32),
                        ft.Text("فایل زمینه موجود نیست", color=DANGER, size=10, text_align=ft.TextAlign.CENTER),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
        )

        return ft.Container(
            width=150,
            padding=8,
            bgcolor=PANEL,
            border=ft.Border.all(2, GOLD if selected_template["name"] == template_name else BORDER),
            border_radius=18,
            ink=True,
            on_click=(lambda e, name=template_name: select_template(name)) if exists else None,
            content=ft.Column(
                [
                    image_control,
                    ft.Text(
                        template_name,
                        size=13,
                        weight=ft.FontWeight.BOLD,
                        color=CREAM,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        "انتخاب این زمینه",
                        size=10,
                        color=GOLD if exists else DANGER,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=7,
            ),
        )

    def has_draft():
        state = read_json(draft_path, {})
        return isinstance(state, dict) and bool(state.get("template"))

    def show_template_selection():
        page.navigation_bar.visible = True
        page.navigation_bar.selected_index = 0
        last_name = app_state.get("last_template")
        draft_name = read_json(draft_path, {}).get("template")

        restore_button = ft.Button(
            content="بازیابی آخرین پیش‌نویس",
            icon=ft.Icons.RESTORE,
            on_click=restore_draft,
            visible=has_draft(),
            style=ft.ButtonStyle(bgcolor=PANEL_2, color=CREAM),
        )

        recent_text = (
            f"آخرین زمینه انتخاب‌شده: {last_name}"
            if last_name in template_paths
            else "هر بار می‌توانید زمینه را آزادانه انتخاب کنید."
        )
        draft_text = (
            f"پیش‌نویس آماده بازیابی است: {draft_name}"
            if draft_name in template_paths
            else ""
        )

        guide_box = card_container(
            ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.LAYERS_OUTLINED, color=GOLD, size=22),
                            ft.Text(
                                "مرحله ۱ — انتخاب زمینه",
                                size=17,
                                weight=ft.FontWeight.BOLD,
                                color=CREAM,
                            ),
                        ],
                        spacing=8,
                    ),
                    ft.Text(
                        "قبل از ورود به اطلاعات جلسه، یکی از هفت زمینه را انتخاب کنید. همهٔ زمینه‌ها از همان مختصات ثابت برای جانمایی متن و عکس استفاده می‌کنند.",
                        size=11,
                        color=MUTED,
                        text_align=ft.TextAlign.RIGHT,
                    ),
                    ft.Text(recent_text, size=10, color=SUCCESS if last_name else MUTED),
                    ft.Text(draft_text, size=10, color=GOLD if draft_text else MUTED),
                ],
                spacing=8,
            ),
            padding=13,
        )

        # پیش‌نمایش زنده از همان صفحه آغاز برنامه هم وجود دارد؛
        # هنوز فرم اطلاعات باز نشده و فقط نمونهٔ زمینه با مقادیر پایه دیده می‌شود.
        preview_template_path = None
        preferred_preview_name = last_name if last_name in template_paths else TEMPLATE_FILES[0][0]
        try:
            candidate = read_template_path(preferred_preview_name)
            preview_template_path = candidate
        except Exception:
            for fallback_name, _filename in TEMPLATE_FILES:
                try:
                    preview_template_path = read_template_path(fallback_name)
                    break
                except Exception:
                    continue

        preview_data = {
            "number": current_next_session_text(),
            "title": "جلسه شنبه های فقهی",
            "subject": "",
            "presenter": "",
            "date": today_jalali_text(),
        }
        if preview_template_path:
            render_live_preview(
                template_path_override=preview_template_path,
                data_override=preview_data,
                update_current=False,
            )

        live_preview_card = card_container(
            ft.Column(
                [
                    section_title(
                        ft.Icons.PREVIEW,
                        "پیش‌نمایش زنده",
                        "نمونهٔ زمینه و مقادیر پیش‌فرض از همان ابتدای برنامه دیده می‌شود.",
                    ),
                    ft.Row([preview_stack], alignment=ft.MainAxisAlignment.CENTER),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=9,
            ),
            padding=12,
            radius=18,
        )

        screen = ft.Column(
            [
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=14, vertical=18),
                    gradient=ft.LinearGradient(
                        begin=ft.Alignment.TOP_LEFT,
                        end=ft.Alignment.BOTTOM_RIGHT,
                        colors=[PANEL, BG],
                    ),
                    border_radius=0,
                    content=ft.Column(
                        [
                            ft.Text(
                                "پوستر ساز جلسات فقهی",
                                size=26,
                                weight=ft.FontWeight.BOLD,
                                color=CREAM,
                                text_align=ft.TextAlign.RIGHT,
                            ),
                            ft.Text(
                                "طراحی سریع، دقیق و حرفه‌ای پوسترهای جلسات فقهی",
                                size=12,
                                color=MUTED,
                                text_align=ft.TextAlign.RIGHT,
                            ),
                            ft.Container(height=6),
                            ft.Row(
                                [
                                    ft.Container(
                                        padding=ft.Padding.symmetric(horizontal=9, vertical=5),
                                        bgcolor=BLACK_GLASS,
                                        border_radius=12,
                                        content=ft.Text(f"{to_persian_digits(TEMPLATE_COUNT)} زمینه آماده", size=10, color=GOLD),
                                    ),
                                    ft.Container(
                                        padding=ft.Padding.symmetric(horizontal=9, vertical=5),
                                        bgcolor=BLACK_GLASS,
                                        border_radius=12,
                                        content=ft.Text("خروجی ۸۴۰×۱۲۶۰", size=10, color=CREAM),
                                    ),
                                ],
                                spacing=7,
                            ),
                        ],
                        spacing=3,
                    ),
                ),
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=12, vertical=14),
                    content=ft.Column(
                        [
                            guide_box,
                            live_preview_card,
                            ft.Row(
                                [template_card(name) for name, _ in TEMPLATE_FILES],
                                wrap=True,
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=10,
                                run_spacing=10,
                            ),
                            restore_button,
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=12,
                    ),
                ),
            ],
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        page_content.content = screen
        page.update()

    # ========================================================
    # صفحه راهنما
    # ========================================================
    def show_help():
        page.navigation_bar.visible = True
        page.navigation_bar.selected_index = 1
        help_items = [
            (
                ft.Icons.LAYERS_OUTLINED,
                "۱. ابتدا زمینه را انتخاب کنید",
                "تا زمانی که یکی از هفت زمینه انتخاب نشود، فرم اطلاعات باز نمی‌شود. با دکمه برگشت در صفحه ویرایش می‌توانید زمینه را عوض کنید.",
            ),
            (
                ft.Icons.EDIT,
                "۲. اطلاعات جلسه را وارد کنید",
                "شماره جلسه بر اساس آخرین پوستر ساخته‌شده پیشنهاد می‌شود؛ پس از هر ساخت موفق، شمارهٔ بعدی خودکار برای جلسه بعد پیشنهاد خواهد شد و دکمه جداگانه‌ای لازم نیست. عنوان «جلسه شنبه های فقهی» و تاریخ امروز هم به‌طور پیش‌فرض قرار دارند.",
            ),
            (
                ft.Icons.CROP,
                "۳. عکس را تنظیم کنید",
                "پس از انتخاب عکس، کادر دقیق ۳:۴ باز می‌شود. با یک انگشت یا موس عکس را جابه‌جا کنید، با +/− یا نوار زوم بزرگنمایی کنید و روی موبایل با دو انگشت زوم و حرکت هم‌زمان انجام دهید.",
            ),
            (
                ft.Icons.AUTO_FIX_HIGH,
                "۴. حذف زمینه",
                "در صفحه تنظیم عکس می‌توانید زمینه را با موتور محلی حذف کنید. در اولین استفاده، مدل سبک u2netp خودکار دانلود و در حافظه خصوصی برنامه ذخیره می‌شود؛ کاربر نباید هیچ فایل یا اسکریپتی اجرا کند. پس از آن پردازش روی همان دستگاه انجام می‌شود.",
            ),
            (
                ft.Icons.PREVIEW,
                "۵. پیش‌نمایش و خروجی",
                "پیش‌نمایش پوستر با هر تغییر متن به‌روزرسانی می‌شود. دکمه ساخت و ذخیره، فایل نهایی JPEG را در پوشه output ذخیره می‌کند.",
            ),
            (
                ft.Icons.RESTORE,
                "۶. پیش‌نویس خودکار",
                "متن‌ها و عکس تأییدشدهٔ فعلی به‌صورت خودکار ذخیره می‌شوند. در صفحه انتخاب زمینه، دکمه «بازیابی آخرین پیش‌نویس» برای ادامه کار قبلی در دسترس است.",
            ),
            (
                ft.Icons.CONTENT_COPY,
                "۷. کپی سریع متن",
                "با «کپی متن جلسه» متن آماده انتشار شامل عنوان، موضوع، ارائه‌دهنده، تاریخ و لینک‌های مجموعه در کلیپ‌بورد قرار می‌گیرد.",
            ),
        ]
        cards = []
        for icon, title, body in help_items:
            cards.append(
                card_container(
                    ft.Row(
                        [
                            ft.Container(
                                width=38,
                                height=38,
                                bgcolor=BLACK_GLASS,
                                border_radius=12,
                                alignment=ft.Alignment.CENTER,
                                content=ft.Icon(icon, color=GOLD, size=21),
                            ),
                            ft.Column(
                                [
                                    ft.Text(title, size=14, weight=ft.FontWeight.BOLD, color=CREAM),
                                    ft.Text(body, size=10, color=MUTED, text_align=ft.TextAlign.RIGHT),
                                ],
                                spacing=4,
                                expand=True,
                            ),
                        ],
                        spacing=10,
                    ),
                    padding=12,
                    radius=15,
                )
            )

        page_content.content = ft.Column(
            [
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=14, vertical=16),
                    content=ft.Column(
                        [
                            ft.Text("راهنمای کامل برنامه", size=23, weight=ft.FontWeight.BOLD, color=CREAM),
                            ft.Text("تمام مسیر ساخت یک پوستر از انتخاب زمینه تا خروجی نهایی", size=11, color=MUTED),
                        ],
                        spacing=2,
                    ),
                ),
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                    content=ft.Column(cards, spacing=10),
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
        )
        page.update()

    # ========================================================
    # صفحه درباره
    # ========================================================
    def show_about():
        page.navigation_bar.visible = True
        page.navigation_bar.selected_index = 2
        page_content.content = ft.Column(
            [
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=14, vertical=18),
                    gradient=ft.LinearGradient(
                        begin=ft.Alignment.TOP_CENTER,
                        end=ft.Alignment.BOTTOM_CENTER,
                        colors=[PANEL, BG],
                    ),
                    content=ft.Column(
                        [
                            ft.Icon(ft.Icons.AUTO_AWESOME, color=GOLD, size=42),
                            ft.Text("پوستر ساز جلسات فقهی", size=24, weight=ft.FontWeight.BOLD, color=CREAM, text_align=ft.TextAlign.CENTER),
                            ft.Text(APP_VERSION, size=11, color=GOLD, text_align=ft.TextAlign.CENTER),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=5,
                    ),
                ),
                ft.Container(
                    padding=12,
                    content=ft.Column(
                        [
                            card_container(
                                ft.Column(
                                    [
                                        ft.Text("هسته برنامه", size=15, weight=ft.FontWeight.BOLD, color=CREAM),
                                        ft.Text("Flet 1.0 + Pillow + پردازش راست‌به‌چپ فارسی", size=11, color=MUTED),
                                    ],
                                    spacing=5,
                                )
                            ),
                            card_container(
                                ft.Column(
                                    [
                                        ft.Text("ساختار خروجی", size=15, weight=ft.FontWeight.BOLD, color=CREAM),
                                        ft.Text("JPEG با کیفیت ۹۵٪، اندازه نهایی ۸۴۰×۱۲۶۰ و عکس ارائه‌دهنده ۶۰۰×۸۰۰ پس از کراپ.", size=11, color=MUTED),
                                    ],
                                    spacing=5,
                                )
                            ),
                            card_container(
                                ft.Column(
                                    [
                                        ft.Text("نکته توسعه", size=15, weight=ft.FontWeight.BOLD, color=CREAM),
                                        ft.Text("برای استفاده از همهٔ زمینه‌ها، فایل‌های template_1.jpg تا template_7.jpg را در assets قرار دهید؛ مختصات متن و عکس در همهٔ زمینه‌ها مشترک است.", size=11, color=MUTED),
                                    ],
                                    spacing=5,
                                )
                            ),
                        ],
                        spacing=10,
                    ),
                ),
            ],
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
        page.update()

    # ========================================================
    # صفحه اصلی ویرایشگر
    # ========================================================
    def show_main_editor():
        if not selected_template["path"]:
            show_template_selection()
            return

        page.navigation_bar.visible = True
        page.navigation_bar.selected_index = 0

        template_name = selected_template["name"] or "زمینه"
        header = ft.Container(
            padding=ft.Padding.symmetric(horizontal=12, vertical=12),
            border=ft.Border.only(bottom=ft.BorderSide(width=1, color="#1C3430")),
            content=ft.Row(
                [
                    ft.Button(
                        content="زمینه‌ها",
                        icon=ft.Icons.ARROW_BACK,
                        on_click=lambda e: show_template_selection(),
                        style=ft.ButtonStyle(bgcolor=PANEL_2, color=CREAM),
                    ),
                    ft.Column(
                        [
                            ft.Text("ویرایش پوستر", size=20, weight=ft.FontWeight.BOLD, color=CREAM),
                            ft.Text(f"زمینه فعال: {template_name}", size=10, color=GOLD),
                        ],
                        spacing=1,
                        expand=True,
                    ),
                    ft.Icon(ft.Icons.EDIT, color=GOLD, size=24),
                ],
                spacing=9,
            ),
        )

        form_panel = card_container(
            ft.Column(
                [
                    section_title(ft.Icons.EDIT, "مشخصات جلسه", "اطلاعات را وارد کنید؛ پیش‌نمایش هم‌زمان به‌روز می‌شود."),
                    txt_number,
                    btn_clear,
                    txt_title,
                    ft.Container(
                        padding=ft.Padding.symmetric(horizontal=2),
                        content=ft.Text(
                            "عنوان پیش‌فرض: جلسه شنبه های فقهی",
                            color=MUTED,
                            size=9,
                            text_align=ft.TextAlign.RIGHT,
                        ),
                    ),
                    txt_subject,
                    ft.Row([ft.Container(expand=True), subject_count]),
                    txt_presenter,
                    ft.Row([ft.Container(expand=True), presenter_count]),
                    txt_date,
                    ft.Row(
                        [
                            ft.Container(expand=True, content=btn_today),
                            ft.Container(expand=True, content=btn_copy_text),
                        ],
                        spacing=8,
                    ),
                    card_container(
                        ft.Column(
                            [
                                ft.Text("عکس ارائه‌دهنده", size=12, color=CREAM, weight=ft.FontWeight.BOLD),
                                ft.Text(
                                    "یک عکس ثابت انتخاب کنید؛ مرحله تنظیم حرفه‌ای عکس بلافاصله باز می‌شود.",
                                    size=9,
                                    color=MUTED,
                                ),
                                btn_select_photo,
                                txt_photo_status,
                            ],
                            spacing=8,
                        ),
                        padding=10,
                        bgcolor="#122821",
                        radius=13,
                    ),
                    btn_generate,
                    status_text,
                ],
                spacing=9,
            ),
            padding=13,
            radius=19,
        )

        preview_panel = card_container(
            ft.Column(
                [
                    section_title(ft.Icons.PREVIEW, "پیش‌نمایش", "همه تغییرات متن و عکس اینجا دیده می‌شوند."),
                    ft.Row([preview_stack], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Row(
                        [
                            ft.Container(expand=True, content=btn_save_as),
                            ft.Container(expand=True, content=btn_share),
                        ],
                        spacing=8,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
            padding=13,
            radius=19,
        )

        workflow = card_container(
            ft.Row(
                [
                    ft.Text("۱ زمینه", size=10, color=GOLD),
                    ft.Text("→", size=11, color=MUTED),
                    ft.Text("۲ اطلاعات", size=10, color=CREAM),
                    ft.Text("→", size=11, color=MUTED),
                    ft.Text("۳ عکس", size=10, color=CREAM),
                    ft.Text("→", size=11, color=MUTED),
                    ft.Text("۴ خروجی", size=10, color=CREAM),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=6,
            ),
            padding=8,
            radius=12,
            bgcolor="#0E211C",
        )

        page_content.content = ft.Column(
            [
                header,
                ft.Container(
                    padding=ft.Padding.symmetric(horizontal=10, vertical=10),
                    content=ft.Column(
                        [workflow, preview_panel, form_panel],
                        spacing=11,
                    ),
                ),
            ],
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
        )
        page.update()

    # ========================================================
    # نوار پیمایش
    # ========================================================
    def nav_changed(e):
        index = int(getattr(e.control, "selected_index", 0) or 0)
        if index == 0:
            if selected_template["path"]:
                show_main_editor()
            else:
                show_template_selection()
        elif index == 1:
            show_help()
        else:
            show_about()

    page.navigation_bar = ft.NavigationBar(
        selected_index=0,
        height=70,
        bgcolor=PANEL,
        indicator_color=BLACK_GLASS,
        destinations=[
            ft.NavigationBarDestination(
                icon=ft.Icons.IMAGE_OUTLINED,
                selected_icon=ft.Icons.IMAGE,
                label="ساخت پوستر",
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.HELP_OUTLINE,
                selected_icon=ft.Icons.HELP,
                label="راهنما",
            ),
            ft.NavigationBarDestination(
                icon=ft.Icons.INFO_OUTLINE,
                selected_icon=ft.Icons.INFO,
                label="درباره",
            ),
        ],
        on_change=nav_changed,
    )

    page.add(page_content)
    update_counts()

    # صفحه شروع: کاربر ابتدا حتماً زمینه را انتخاب می‌کند.
    show_template_selection()


if __name__ == "__main__":
    ft.run(main, assets_dir="assets")


