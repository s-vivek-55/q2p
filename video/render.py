import argparse
import json
import math
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat

from audio import AUDIO, SAMPLE_RATE, load_storyboard


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
PREVIEWS = ROOT / "previews_revised"
SEGMENTS = ROOT / "segments_revised"
WIDTH, HEIGHT, FPS = 1920, 1080, 30
NAVY = "#0A101D"
WHITE = "#F3F7FC"
MUTED = "#ACBDCE"
CYAN = "#4DD9F1"
GREEN = "#4AE2AE"
AMBER = "#FFCA7A"
LINE = "#2A3B51"
PANEL = "#101B2B"
STORYBOARD = load_storyboard()
ACT_STARTS = [min(scene["start"] for scene in STORYBOARD["scenes"] if scene["act"] == act) for act in range(1, 7)] + [STORYBOARD["duration"]]
ACT_LABELS = ["01  Why the wait?", "02  How it works", "03  Lumen APIs", "04  Existing customer", "05  New customer", "06  Lumen benefits"]
FONT_PATHS = {
    "display": "C:/Windows/Fonts/bahnschrift.ttf",
    "body": "C:/Windows/Fonts/segoeui.ttf",
    "bold": "C:/Windows/Fonts/segoeuib.ttf",
    "mono": "C:/Windows/Fonts/consola.ttf",
}


@lru_cache(maxsize=150)
def font(size: int, family: str = "body") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATHS[family], size)


@lru_cache(maxsize=30)
def asset(name: str) -> Image.Image:
    return Image.open(ASSETS / name).convert("RGB")


@lru_cache(maxsize=100)
def cropped_asset(name: str, box: tuple) -> Image.Image:
    image = asset(name)
    assert 0 <= box[0] < box[2] <= image.width, (name, box)
    assert 0 <= box[1] < box[3] <= image.height, (name, box)
    return image.crop(box)


@lru_cache(maxsize=20)
def narration_cues(identifier, prefixes):
    boundaries = [json.loads(line) for line in (AUDIO / f"{identifier}.jsonl").read_text(encoding="utf-8").splitlines()]
    first_offset = boundaries[0]["offset"]
    cues = []
    for prefix in prefixes:
        matches = [boundary for boundary in boundaries if boundary["text"].casefold().startswith(prefix.casefold())]
        assert matches, (identifier, prefix)
        cues.append(0.25 + (matches[0]["offset"] - first_offset) / 1e7)
    assert cues == sorted(cues)
    return cues


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


def background() -> Image.Image:
    pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for row in range(HEIGHT):
        factor = row / HEIGHT
        pixels[row, :, :] = (10 + int(2 * factor), 16 + int(5 * factor), 29 + int(7 * factor))
    image = Image.fromarray(pixels)
    drawing = ImageDraw.Draw(image)
    for column in range(64, WIDTH, 128):
        drawing.line((column, 126, column, 972), fill="#0F1A2A", width=1)
    for row in range(152, 982, 128):
        drawing.line((66, row, 1860, row), fill="#0F1A2A", width=1)
    return image


BACKGROUND = background()
LOGO_MASK = Image.open(ASSETS / "image3.png").convert("L").resize((222, 59), Image.Resampling.LANCZOS)


class Canvas:
    def __init__(self):
        self.image = BACKGROUND.copy()
        self.drawing = ImageDraw.Draw(self.image)

    def text(self, value, left, top, size=36, color=WHITE, family="body", max_width=None):
        value = str(value)
        lines = value.split("\n")
        max_width = WIDTH - left - 88 if max_width is None else max_width
        chosen_size = size
        while max(font(chosen_size, family).getlength(line) for line in lines) > max_width:
            chosen_size -= 1
            assert chosen_size >= 18, f"Text cannot fit: {value}"
        selected = font(chosen_size, family)
        for index, line in enumerate(lines):
            baseline_top = top + index * round(chosen_size * 1.22)
            bounds = self.drawing.textbbox((left, baseline_top), line, font=selected, anchor="lt")
            assert bounds[0] >= 0 and bounds[2] <= WIDTH and bounds[3] <= HEIGHT, (value, bounds)
            self.drawing.text((left, baseline_top), line, font=selected, fill=color, anchor="lt")

    def paragraph(self, value, left, top, width, size=32, color=MUTED):
        lines = []
        current = ""
        for word in value.split():
            candidate = f"{current} {word}".strip()
            if font(size).getlength(candidate) > width and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        self.text("\n".join(lines), left, top, size, color, max_width=width)

    def box(self, box, fill=PANEL, outline=LINE, radius=8, width=2):
        self.drawing.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def arrow(self, start, end, color=CYAN, width=3):
        self.drawing.line((*start, *end), fill=color, width=width)
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        points = [end]
        for offset in (0.45, -0.45):
            points.append((end[0] - 14 * math.cos(angle + offset), end[1] - 14 * math.sin(angle + offset)))
        self.drawing.polygon(points, fill=color)

    def glow(self, box, color=CYAN, opacity=1.0):
        overlay = ImageDraw.Draw(self.image, "RGBA")
        red, green, blue = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        overlay.rounded_rectangle(box, radius=7, fill=(red, green, blue, int(14 * opacity)), outline=(red, green, blue, int(220 * opacity)), width=3)

    def chip(self, text, left, top, width, color=CYAN):
        self.box((left, top, left + width, top + 58), outline=color)
        self.text(text, left + 20, top + 16, 26, color, "bold", width - 40)

    def reveal(self, mapped, box, local_time, start, duration, fill, typing=False):
        amount = ease((local_time - start) / duration)
        if amount >= 1:
            return
        overlay = ImageDraw.Draw(self.image, "RGBA")
        if typing:
            box = (box[0] + (box[2] - box[0]) * amount, box[1], box[2], box[3])
            opacity = 255
        else:
            opacity = round(255 * (1 - amount))
        overlay.rectangle(mapped(box), fill=(*fill, opacity))

    def metric(self, value, label, left, top, width=400, color=GREEN, size=78):
        self.text(value, left, top, size, color, "display", width)
        self.text(label, left, top + size + 17, 27, MUTED, max_width=width)

    def shot(self, name, source, target, local_time, duration):
        original = cropped_asset(name, tuple(source))
        left, top, width, height = target
        scale = min(width / original.width, height / original.height)
        shown_width = round(original.width * scale)
        shown_height = round(original.height * scale)
        left += (width - shown_width) // 2
        top += (height - shown_height) // 2
        zoom = 1.014 - 0.010 * local_time / duration
        horizontal_scale = shown_width * zoom / original.width
        vertical_scale = shown_height * zoom / original.height
        offset_left = shown_width * (zoom - 1) / 2
        offset_top = shown_height * (zoom - 1) / 2
        zoomed = original.transform(
            (shown_width, shown_height),
            Image.Transform.AFFINE,
            (1 / horizontal_scale, 0, offset_left / horizontal_scale, 0, 1 / vertical_scale, offset_top / vertical_scale),
            Image.Resampling.BICUBIC,
        )
        self.image.paste(zoomed, (left, top))
        self.drawing.rounded_rectangle((left - 2, top - 2, left + shown_width + 1, top + shown_height + 1), radius=6, outline=LINE, width=2)

        def mapped(box):
            return (
                left + (box[0] - source[0]) * horizontal_scale - offset_left,
                top + (box[1] - source[1]) * vertical_scale - offset_top,
                left + (box[2] - source[0]) * horizontal_scale - offset_left,
                top + (box[3] - source[1]) * vertical_scale - offset_top,
            )

        return mapped

    def cursor(self, left, top, local_time, click_time):
        travel = ease(local_time / max(click_time, 0.1))
        left += (1 - travel) * 160
        top += (1 - travel) * 100
        points = [(left, top), (left, top + 43), (left + 12, top + 31), (left + 23, top + 50), (left + 32, top + 45), (left + 21, top + 26), (left + 40, top + 25)]
        self.drawing.polygon(points, fill=WHITE, outline="#162035", width=2)
        phase = local_time - click_time
        if 0 <= phase < 0.75:
            radius = 16 + 58 * phase / 0.75
            overlay = ImageDraw.Draw(self.image, "RGBA")
            overlay.ellipse((left - radius, top - radius, left + radius, top + radius), outline=(74, 226, 174, round(230 * (1 - phase / 0.75))), width=4)


def chrome(canvas, scene, local_time):
    canvas.text("CONNECTIQ  /  EXECUTIVE DEMO", 88, 61, 23, CYAN, "bold")
    canvas.image.paste(WHITE, (1606, 39, 1828, 98), LOGO_MASK)
    canvas.drawing.line((88, 117, 1832, 117), fill=LINE, width=1)
    canvas.text(scene["eyebrow"], 88, 158, 25, CYAN, "bold")
    canvas.text(scene["title"], 88, 206, 55, WHITE, "display", 1744)
    heading_length = min(1744, font(55, "display").getlength(scene["title"]))
    canvas.drawing.line((88, 282, 1832, 282), fill=LINE, width=1)
    canvas.drawing.line((88, 282, 88 + heading_length * ease(local_time / 0.55), 282), fill=GREEN if scene["act"] == 6 else CYAN, width=4)
    position = scene["start"] + local_time
    for index, label in enumerate(ACT_LABELS):
        left = 88 + round(ACT_STARTS[index] / STORYBOARD["duration"] * 1744)
        right = 88 + round(ACT_STARTS[index + 1] / STORYBOARD["duration"] * 1744) - 10
        canvas.drawing.line((left, 988, right, 988), fill=LINE, width=3)
        amount = ease((position - ACT_STARTS[index]) / max(ACT_STARTS[index + 1] - ACT_STARTS[index], 1))
        if position > ACT_STARTS[index]:
            canvas.drawing.line((left, 988, left + (right - left) * amount, 988), fill=GREEN if index >= 3 else CYAN, width=3)
        canvas.text(label, left, 1011, 20, WHITE if index + 1 == scene["act"] else MUTED, max_width=right - left)
    if 3 <= scene["act"] <= 5:
        canvas.text("DEMO ONLY / NO LIVE CONNECTIONS OR BILLING", 1056, 64, 20, MUTED, max_width=480)


def fibers(canvas, local_time):
    drawing = ImageDraw.Draw(canvas.image, "RGBA")
    for lane in range(11):
        points = []
        for step in range(161):
            horizontal = step * 12
            vertical = 787 + lane * 12 - 205 * math.exp(-((horizontal - 1570) / 520) ** 2) + 22 * math.sin(horizontal / 310 + lane * 0.15)
            points.append((horizontal, vertical))
        drawing.line(points, fill=(43, 116, 155, 65), width=2)
        head = int((local_time * 17 + lane * 13) % len(points))
        for tail in range(12):
            segment = head - tail
            if 0 < segment < len(points):
                drawing.line((points[segment - 1], points[segment]), fill=(65, 210, 239, max(0, 180 - tail * 15)), width=3)


def flow_node(canvas, number, label, left, top, width, active=False, color=CYAN):
    canvas.box((left, top, left + width, top + 156), outline=color if active else LINE)
    canvas.text(number, left + 24, top + 20, 26, color, "bold")
    canvas.text(label, left + 24, top + 67, 34, WHITE, "display", width - 48)


def render_scene(scene, local_time):
    canvas = Canvas()
    identifier = scene["id"]
    duration = scene["duration"]
    progress = ease(local_time / duration)
    if identifier == "close":
        fibers(canvas, local_time)
        message = "Thank you."
        left = (WIDTH - font(132, "display").getlength(message)) / 2
        canvas.text(message, left, 423, 132, WHITE, "display", 1100)
        canvas.drawing.line((822, 622, 1098, 622), fill=GREEN, width=4)
        return canvas.image
    if identifier == "title":
        fibers(canvas, local_time)
        canvas.image.paste(WHITE, (1534, 76, 1828, 154), LOGO_MASK.resize((294, 78), Image.Resampling.LANCZOS))
        canvas.text("TEAM CONNECTIQ", 104, 102, 29, CYAN, "bold")
        offset = round(24 * (1 - ease(local_time / 1.1)))
        canvas.text("Lumen Zero-Touch", 104, 278 + offset, 108, WHITE, "display", 1690)
        canvas.text("NaaS Broker", 104, 408 + offset, 108, WHITE, "display", 1690)
        canvas.text("The network is ready.", 109, 588, 46, MUTED)
        canvas.text("Let's remove the wait.", 109, 647, 53, CYAN, "display")
        canvas.text("LESS MANUAL WORK. EARLIER REVENUE FOR LUMEN.", 108, 922, 27, WHITE, "bold")
        return canvas.image
    chrome(canvas, scene, local_time)

    if identifier == "foundation":
        canvas.text("BUILDINGS ALREADY CONNECTED TO LUMEN", 112, 321, 27, MUTED, "bold", 700)
        for building_index, (left, height) in enumerate([(150, 244), (320, 328), (507, 275)]):
            top = 750 - height
            canvas.box((left, top, left + 136, 750), fill="#132337", outline=CYAN if building_index == int(local_time / 4) % 3 else LINE, radius=1)
            for floor in range(int(height / 48) - 1):
                for column in range(3):
                    window_left = left + 21 + column * 35
                    window_top = top + 26 + floor * 48
                    canvas.drawing.rectangle((window_left, window_top, window_left + 15, window_top + 18), fill="#284B63")
        canvas.drawing.line((128, 777, 699, 777), fill=CYAN, width=4)
        canvas.arrow((710, 585), (862, 585), CYAN, 4)
        for index, (title, subtitle) in enumerate([
            ("Internet-on-Demand", "Switch capacity on when it is needed"),
            ("Cloud Connect", "Connect to AWS, Google Cloud or Azure"),
            ("One API call", "A software instruction starts activation"),
        ]):
            top = 345 + index * 170
            canvas.drawing.line((923, top + 7, 923, top + 106), fill=GREEN if index == 2 else CYAN, width=4)
            canvas.text(title, 957, top, 45, WHITE, "display", 820)
            canvas.text(subtitle, 959, top + 65, 29, MUTED, max_width=820)
        if local_time >= 9:
            canvas.glow((925, 676, 1820, 810), GREEN)
        canvas.metric("$0 CapEx", "No new construction cost where eligible fiber is already installed.", 137, 823, 1650, GREEN, 64)

    elif identifier == "relay":
        labels = ["Read\nemail", "Check local\nservices", "Choose\ncloud", "Prepare\nquote", "Send\norder"]
        for index, label in enumerate(labels):
            left = 110 + index * 346
            flow_node(canvas, f"0{index + 1}", label, left, 374, 310, min(4, int(local_time / duration * 5)) == index)
            if index < 4:
                canvas.arrow((left + 315, 452), (left + 340, 452), MUTED, 2)
        canvas.text("An agent looks up services, checks prices and types the order.", 112, 611, 37, MUTED, max_width=1690)
        canvas.metric("1-3 weeks", "Example of a manual quoting delay", 112, 745, 780, AMBER, 108)
        canvas.text("The fiber is ready.\nRevenue has to wait.", 1034, 754, 51, WHITE, "display", 770)

    elif identifier == "friction":
        pain_points = ["Emails need manual review", "Quotes can take 1-3 weeks", "Customers cannot order directly", "Typing can create wrong orders", "Some customers leave while waiting"]
        for index, label in enumerate(pain_points):
            top = 336 + index * 110
            active = local_time >= index * 1.25
            canvas.text(f"0{index + 1}", 111, top, 31, AMBER if active else LINE, "display")
            canvas.text(label, 187, top + 1, 35, WHITE if active else MUTED, max_width=870)
            canvas.drawing.line((186, top + 73, 974, top + 73), fill=LINE, width=1)
        canvas.text("READY CAPACITY", 1118, 370, 28, MUTED, "bold")
        canvas.text("UNBILLED", 1108, 441, 101, AMBER, "display", 700)
        canvas.text("Time before billing can start", 1120, 592, 34, WHITE, max_width=680)
        for index in range(21):
            left = 1120 + index * 31
            canvas.drawing.rectangle((left, 686, left + 23, 788), fill=AMBER if index < round(21 * progress) else LINE)
        canvas.text("DAY 01", 1120, 818, 25, MUTED)
        canvas.text("DAY 21", 1660, 818, 25, MUTED)

    elif identifier == "scope":
        for index, (title, journey, image_name, crop) in enumerate([
            ("Existing customer", "Chat  >  Quote  >  Approve", "image16.png", (269, 230, 1173, 418)),
            ("New customer", "Describe  >  Compare  >  Order", "image21.png", (262, 158, 856, 296)),
        ]):
            top = 336 + index * 238
            canvas.box((110, top, 1810, top + 197), outline=CYAN if index == 0 else GREEN)
            canvas.text(f"0{index + 1}", 141, top + 28, 28, CYAN if index == 0 else GREEN, "bold")
            canvas.text(title, 219, top + 30, 46, WHITE, "display", 770)
            canvas.text(journey, 222, top + 103, 30, MUTED, max_width=740)
            canvas.shot(image_name, crop, (1036, top + 23, 737, 151), local_time, duration)
        canvas.text("Describe the need. The broker prepares the order.", 112, 866, 48, GREEN, "display", 1700)
        canvas.text("This prototype simulates activation and billing; it does not create real charges.", 114, 931, 23, MUTED)

    elif identifier == "plan":
        labels = ["Read the\nrequest", "Check\navailability", "Calculate\nthe price", "Show the\nquote"]
        for index, label in enumerate(labels):
            left = 110 + index * 435
            flow_node(canvas, f"0{index + 1}", label, left, 358, 389, index == min(3, int(local_time / 3.1)), GREEN if index == 2 else CYAN)
            if index < 3:
                canvas.arrow((left + 396, 436), (left + 427, 436))
        canvas.text("Check spare capacity. Compare cloud options. Calculate using fixed rates.", 115, 629, 34, MUTED, max_width=1690)
        canvas.box((110, 740, 1810, 907), fill="#102A27", outline=GREEN)
        canvas.text("AI never invents a price.", 155, 784, 65, GREEN, "display", 1570)

    elif identifier == "gate":
        labels = ["Approve quote", "Check order", "Send API request", "Activate service"]
        for index, label in enumerate(labels):
            left = 110 + index * 435
            flow_node(canvas, f"0{index + 1}", label, left, 416, 389, local_time > index * 1.45, GREEN)
            if index < 3:
                canvas.arrow((left + 396, 494), (left + 427, 494), GREEN)
        canvas.text("Nothing is ordered until the customer approves.", 112, 731, 58, WHITE, "display", 1690)
        canvas.text("The order is checked against the agreed location, speed, cloud and price.", 116, 829, 34, MUTED, max_width=1680)
        canvas.text("THIS DEMO SIMULATES SERVICE ACTIVATION AND BILLING", 115, 920, 22, CYAN, "bold")

    elif identifier == "portal":
        mapped = canvas.shot("image13.png", (0, 69, 310, 392), (112, 304, 626, 650), local_time, duration)
        labels = [("Multi-Cloud Gateway", (6, 143, 292, 178)), ("Order > Ordering", (7, 187, 293, 252)), ("Quote", (7, 262, 293, 291)), ("Service Details", (7, 332, 293, 367))]
        phase = min(3, int(max(0, local_time - 0.3) / 1.5))
        canvas.glow(mapped(labels[phase][1]))
        canvas.text("developer.lumen.com", 866, 339, 41, CYAN, "display", 935)
        for index, (label, _) in enumerate(labels):
            canvas.text(f"0{index + 1}", 867, 447 + index * 109, 27, CYAN, "bold")
            canvas.text(label, 941, 441 + index * 109, 43, WHITE if index == phase else MUTED, "display", 846)
        canvas.text("Official Lumen Connect API reference", 869, 914, 28, MUTED, max_width=890)

    elif identifier == "endpoint":
        canvas.text("POST  /Customer/v3/Ordering/orderRequest", 112, 313, 43, CYAN, "mono", 1690)
        mapped = canvas.shot("image13.png", (390, 253, 1437, 616), (110, 401, 1700, 517), local_time, duration)
        box = (398, 259, 1422, 301) if local_time < 2.3 else (410, 502, 954, 596)
        canvas.glow(mapped(box), GREEN if local_time >= 2.3 else CYAN)
        canvas.text("x-customer-number identifies the customer placing the order.", 112, 932, 30, GREEN, "bold", 1690)

    elif identifier == "contract":
        mapped = canvas.shot("image13.png", (415, 786, 762, 1010), (110, 317, 864, 619), local_time, duration)
        fields = [("externalId", "Customer's order reference", (431, 802, 706, 822)), ("billingAccount", "Account that will be billed", (431, 821, 674, 870)), ("channel", "How the order was submitted", (431, 872, 659, 926)), ("quote", "The price the customer approved", (431, 929, 687, 1004))]
        phase = min(3, int(max(0, local_time - 0.3) / 1.35))
        canvas.glow(mapped(fields[phase][2]))
        for index, (label, description, _) in enumerate(fields):
            top = 347 + index * 144
            canvas.text(label, 1103, top, 44, CYAN if index == phase else WHITE, "mono", 695)
            canvas.text(description, 1105, top + 62, 29, MUTED, max_width=690)

    elif identifier == "mock":
        canvas.text("LOCAL MOCK APIS / SAFE TESTING", 116, 331, 28, GREEN, "bold", 770)
        for index, name in enumerate(["Ordering", "Serviceability", "Quoting", "Accounts"]):
            top = 404 + index * 112
            canvas.box((111, top, 797, top + 88))
            canvas.text(name, 143, top + 22, 36, WHITE, "display", 430)
            canvas.text("MOCK", 646, top + 29, 22, GREEN, "bold", 119)
        canvas.arrow((836, 603), (1037, 603), CYAN)
        canvas.text("BEFORE LIVE USE", 1111, 410, 35, WHITE, "bold", 682)
        canvas.paragraph("Connect to Lumen, add secure access, and test real orders before launch.", 1114, 493, 672, 35)
        canvas.text("NO LIVE CONNECTIONS ARE CHANGED. NO REAL BILLS ARE CREATED.", 113, 905, 33, GREEN, "bold", 1690)

    elif identifier == "chat":
        for text, left, width in [("5 Gbps", 110, 220), ("AWS or GCP", 354, 290), ("DDoS protection", 668, 363)]:
            canvas.chip(text, left, 302, width)
        mapped = canvas.shot("image16.png", (266, 226, 1177, 421), (110, 400, 1700, 377), local_time, duration)
        phase = 0 if local_time < 3.6 else 1 if local_time < 5.0 else 2
        rectangles = [(274, 234, 1168, 279), (275, 301, 1139, 341), (274, 368, 1168, 414)]
        canvas.glow(mapped(rectangles[phase]), GREEN if phase == 2 else CYAN)
        canvas.reveal(mapped, (309, 244, 1102, 273), local_time, 0.2, 2.9, asset("image16.png").getpixel((1110, 250)), typing=True)
        canvas.reveal(mapped, (268, 294, 1173, 349), local_time, 3.4, 0.3, asset("image16.png").getpixel((1175, 319)))
        canvas.reveal(mapped, (268, 362, 1173, 419), local_time, 4.8, 0.3, asset("image16.png").getpixel((1175, 355)))
        canvas.text("The assistant asks only for the missing city.", 116, 833, 43, WHITE, "display", 1680)
        if local_time > 4.9:
            canvas.text("DALLAS  +  GOOGLE CLOUD", 117, 908, 30, GREEN, "bold")

    elif identifier == "quote":
        mapped = canvas.shot("image16.png", (1220, 150, 1827, 270), (110, 339, 1700, 354), local_time, duration)
        if local_time < 3.0:
            highlight = (1227, 201, 1407, 262)
        elif local_time < 4.8:
            highlight = (1629, 201, 1823, 262)
        else:
            highlight = None
            canvas.glow((104, 728, 1804, 889), GREEN)
        if highlight is not None:
            canvas.glow(mapped(highlight), GREEN)
        canvas.metric("6 unused 10G ports", "Already available in Dallas. No new construction is needed.", 115, 748, 1645, GREEN, 74)
        canvas.shot("image16.png", (309, 510, 1117, 539), (120, 908, 1644, 58), local_time, duration)

    elif identifier == "bom":
        mapped = canvas.shot("image16.png", (1222, 278, 1828, 491), (110, 309, 1700, 603), local_time, duration)
        phase = min(4, int(max(0, local_time - 0.4) / 1.0))
        canvas.glow(mapped((1233, 341 + phase * 29, 1818, 369 + phase * 29)), GREEN if phase == 4 else CYAN)
        canvas.text("Port  +  cloud connection  +  data transfer  +  attack protection  +  $0 construction", 113, 940, 28, MUTED, max_width=1694)

    elif identifier == "compare":
        mapped = canvas.shot("image16.png", (1238, 551, 1814, 676), (110, 363, 1700, 388), local_time, duration)
        canvas.glow(mapped((1245, 613, 1807, 642)), GREEN)
        canvas.text("GCP selected by the customer", 116, 804, 52, GREEN, "display", 1660)
        canvas.text("These are cloud connection and data-transfer costs, not the total network price.", 117, 892, 28, MUTED, max_width=1680)

    elif identifier == "topology":
        mapped = canvas.shot("image16.png", (1230, 740, 1821, 860), (110, 376, 1700, 389), local_time, duration)
        points = [(1349, 777), (1400, 777), (1431, 807), (1520, 807), (1575, 807), (1678, 808), (1778, 808)]
        phase = (local_time * 0.26) % 1
        step = min(len(points) - 2, int(phase * (len(points) - 1)))
        subphase = phase * (len(points) - 1) - step
        horizontal = points[step][0] + (points[step + 1][0] - points[step][0]) * subphase
        vertical = points[step][1] + (points[step + 1][1] - points[step][1]) * subphase
        mark = mapped((horizontal - 3, vertical - 3, horizontal + 3, vertical + 3))
        canvas.drawing.ellipse(mark, fill=GREEN)
        canvas.text("Dallas office  >  Lumen + attack protection  >  Google Cloud", 113, 847, 37, WHITE, "display", 1690)
        canvas.text("The proposed route is visible before the customer approves.", 115, 920, 27, MUTED)

    elif identifier == "approve":
        canvas.text("The customer reviews the price and chooses whether to proceed.", 113, 330, 38, MUTED, max_width=1690)
        canvas.shot("image17.png", (48, 82, 188, 130), (645, 494, 630, 216), local_time, duration)
        canvas.cursor(1002, 606, local_time, 1.4)
        canvas.text("APPROVE THE QUOTE  >  CHECK THE ORDER  >  SEND IT", 209, 829, 38, GREEN, "bold", 1500)

    elif identifier == "validated":
        canvas.text("SITE AND SPEED", 114, 325, 25, CYAN, "bold")
        canvas.text("APPROVED BILLING", 1080, 325, 25, GREEN, "bold")
        mapped = canvas.shot("image18.png", (30, 292, 494, 516), (110, 386, 872, 489), local_time, duration)
        canvas.shot("image18.png", (40, 700, 515, 820), (1060, 443, 741, 242), local_time, duration)
        canvas.glow(mapped((47, 310, 405, 503)), CYAN)
        canvas.text("Q-989666A8", 1075, 759, 43, GREEN, "mono", 730)
        canvas.text("Checked against the approved quote", 1077, 831, 27, MUTED, max_width=720)

    elif identifier == "success":
        canvas.metric("$6.1548", "per hour / simulated billing", 111, 387, 650, GREEN, 105)
        canvas.text("200 SUCCESS", 116, 598, 43, WHITE, "display", 625)
        mapped = canvas.shot("image19.png", (37, 79, 634, 321), (806, 343, 1002, 438), local_time, duration)
        canvas.glow(mapped((62, 145, 514, 211)), GREEN)
        canvas.text("LUMEN-NAAS-30104", 814, 812, 41, GREEN, "mono", 983)
        for label, left, width in [("Sales quoting", 115, 421), ("Routine solution design", 600, 590), ("Manual order entry", 1282, 521)]:
            canvas.text(label, left, 925, 28, MUTED, max_width=width)
            canvas.drawing.line((left - 3, 941, left + min(width - 10, font(28).getlength(label)) + 4, 941), fill=GREEN, width=2)

    elif identifier == "intake":
        mapped = canvas.shot("image21.png", (258, 150, 861, 300), (111, 357, 1696, 422), local_time, duration)
        for index, box in enumerate([(270, 184, 849, 202), (270, 202, 849, 218), (270, 218, 849, 237)]):
            canvas.reveal(mapped, box, local_time, 0.2 + index * 1.15, 1.2, asset("image21.png").getpixel((849, 231)), typing=True)
        if local_time > 5.8:
            canvas.glow(mapped((263, 264, 449, 297)), GREEN)
            location = mapped((406, 281, 407, 282))
            canvas.cursor(location[0], location[1], local_time - 5.6, 1.2)
        canvas.text("APPLE  /  CHICAGO  /  5 FLOORS  /  100 SQ FT PER FLOOR", 113, 840, 29, CYAN, "bold", 1694)
        canvas.text("AWS + protection from online attacks", 113, 904, 38, WHITE, "display")

    elif identifier == "survey":
        shots = [((263, 309, 856, 384), (111, 317, 1080, 137)), ((263, 463, 856, 595), (111, 471, 1080, 240)), ((263, 673, 856, 770), (111, 728, 1080, 177))]
        mappers = [canvas.shot("image21.png", source, destination, local_time, duration) for source, destination in shots]
        field_color = asset("image21.png").getpixel((780, 363))
        fields = [
            (0, (278, 351, 477, 377), 0.25),
            (1, (278, 490, 492, 516), 0.75),
            (1, (278, 558, 375, 588), 1.25),
            (1, (575, 558, 675, 588), 1.65),
            (2, (278, 698, 492, 726), 2.1),
        ]
        for mapper_index, box, start in fields:
            canvas.reveal(mappers[mapper_index], box, local_time, start, 0.45, field_color)
            if start <= local_time < start + 0.8:
                canvas.glow(mappers[mapper_index](box), GREEN)
        canvas.reveal(mappers[2], (273, 747, 289, 765), local_time, 2.5, 0.4, asset("image21.png").getpixel((300, 749)))
        for index, (value, label) in enumerate([("Apple", "Company"), ("Chicago", "On-net location"), ("5 floors / 500 sq ft", "Site dimensions"), ("AWS + DDoS", "Cloud and protection")]):
            top = 334 + index * 151
            canvas.text(value, 1303, top, 43, GREEN if local_time > index * 1.1 else WHITE, "display", 500)
            canvas.text(label, 1306, top + 65, 27, MUTED, max_width=491)
        canvas.text("AUTO-FILLED. ALWAYS EDITABLE.", 114, 936, 26, CYAN, "bold")

    elif identifier == "sizing":
        mapped = canvas.shot("image21.png", (906, 104, 1807, 216), (111, 303, 1695, 219), local_time, duration)
        metrics = [(910, 145, 1120, 211), (1131, 145, 1346, 211), (1590, 145, 1794, 211)]
        canvas.glow(mapped(metrics[min(2, int(local_time / 2.1))]), GREEN)
        canvas.text("SERVICES AVAILABLE AT THIS LOCATION", 113, 556, 27, CYAN, "bold")
        mapped_services = canvas.shot("image21.png", (909, 368, 1821, 579), (111, 603, 1695, 349), local_time, duration)
        if local_time > 5.8:
            canvas.glow(mapped_services((914, 459, 1815, 489)), CYAN)

    elif identifier == "plans":
        mapped = canvas.shot("image21.png", (1205, 628, 1809, 750), (111, 316, 1695, 365), local_time, duration)
        canvas.glow(mapped((1208, 663, 1800, 693)), GREEN)
        canvas.shot("image21.png", (909, 768, 1820, 837), (111, 704, 1695, 131), local_time, duration)
        for value, label, left, width in [("$1.0848/hr", "AWS us-east-2", 113, 430), ("$791.90/mo", "Estimated monthly", 560, 437), ("$24.60/mo", "Savings vs. highest cloud", 1011, 433), ("$0", "On-net CapEx", 1501, 301)]:
            canvas.metric(value, label, left, 864, width, GREEN, 53)

    elif identifier == "new_quote":
        canvas.shot("image22.png", (911, 26, 1827, 130), (111, 311, 1695, 194), local_time, duration)
        mapped = canvas.shot("image22.png", (911, 151, 1828, 364), (111, 548, 1695, 393), local_time, duration)
        canvas.glow(mapped((918, 304, 1817, 355)), GREEN)

    elif identifier == "onboard":
        canvas.shot("image22.png", (951, 467, 1787, 641), (111, 344, 1695, 356), local_time, duration)
        canvas.text("CREATE ACCOUNT  >  CHECK THE ORDER  >  ORDER THE CONNECTION", 214, 744, 28, CYAN, "bold", 1500)
        canvas.shot("image22.png", (912, 800, 1135, 843), (532, 819, 855, 141), local_time, duration)
        canvas.cursor(1124, 869, local_time, 1.5)

    elif identifier == "revenue":
        canvas.text("DALLAS DEMO RATE  /  $6.1548 PER HOUR", 113, 317, 29, CYAN, "bold")
        value_progress = ease(local_time / 3)
        canvas.metric(f"${round(1034 * value_progress):,}", "Billing delayed by 1 week's wait", 112, 437, 790, GREEN, 133)
        canvas.metric(f"${round(3102 * value_progress):,}", "Billing delayed by 3 weeks' wait", 1043, 437, 768, GREEN, 133)
        canvas.drawing.line((115, 716, 1807, 716), fill=LINE, width=2)
        canvas.arrow((116, 810), (1780, 810), CYAN, 4)
        canvas.text("READY CAPACITY", 116, 753, 26, WHITE, "bold")
        canvas.text("EARLIER RECURRING REVENUE", 1123, 753, 27, GREEN, "bold", 670)
        canvas.text("$6.1548 x 24 hours x 7-21 days, per ready connection", 115, 851, 32, MUTED)
        canvas.text("Example revenue that could be earned sooner; this is not guaranteed profit.", 115, 929, 26, MUTED)

    elif identifier == "drivers":
        entries = [
            ("Earn revenue sooner", "Activate the ready connection and start billing sooner."),
            ("Lower sales costs", "Spend less staff time on routine quotes and data entry."),
            ("Earn more from existing fiber", "Make it easier for customers to buy extra capacity."),
            ("Reduce mistakes and support work", "Check the order details before they are sent."),
            ("Win more new customers", "Show prices immediately, before buyers lose interest."),
        ]
        cues = narration_cues("drivers", ("First", "Second", "Third", "Fourth", "Finally"))
        active_index = max(0, sum(local_time >= cue for cue in cues) - 1)
        for index, (title, subtitle) in enumerate(entries):
            top = 315 + index * 129
            active = index == active_index
            color = GREEN if active else MUTED
            if active:
                canvas.box((102, top - 13, 1811, top + 104), fill="#112B29", outline=GREEN)
            canvas.text(f"0{index + 1}", 113, top + 3, 46, color, "display")
            canvas.text(title, 242, top, 38, GREEN if active else WHITE, "display", 1470)
            canvas.text(subtitle, 245, top + 55, 28, MUTED, max_width=1470)
            canvas.drawing.line((242, top + 104, 1806, top + 104), fill=LINE, width=1)

    elif identifier in ("roadmap_one", "roadmap_two"):
        entries = (
            [("01", "Live Lumen services and prices", "Use current availability and each customer's agreed rates."), ("02", "Secure customer sign-in", "Link the order to the right customer account.")]
            if identifier == "roadmap_one"
            else [("03", "Adjust capacity automatically", "Add bandwidth when demand rises. Reduce it when demand falls."), ("04", "Support and protect the network", "Find problems, manage support tickets and defend against attacks.")]
        )
        canvas.drawing.line((164, 370, 164, 847), fill=LINE, width=3)
        for index, (number, title, subtitle) in enumerate(entries):
            top = 379 + index * 257
            canvas.drawing.ellipse((151, top + 43, 177, top + 69), fill=CYAN if index == 0 else GREEN)
            canvas.text(number, 225, top + 3, 96, CYAN if index == 0 else GREEN, "display", 176)
            canvas.text(title, 432, top + 3, 51, WHITE, "display", 1372)
            canvas.text(subtitle, 437, top + 85, 34, MUTED, max_width=1366)
        canvas.text("FUTURE PLANS: LIVE CONNECTIONS, SECURITY CHECKS AND TESTING ARE STILL REQUIRED", 115, 932, 25, CYAN, "bold", 1686)
    else:
        raise ValueError(f"Unknown scene: {identifier}")
    return canvas.image


def setup_stream(container, width=WIDTH, height=HEIGHT):
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, FPS)
    stream.options = {"preset": "fast", "crf": "18", "profile": "high", "level": "4.1", "bf": "0", "threads": "4"}
    return stream


def encode_frame(container, stream, image, index):
    frame = av.VideoFrame.from_image(image)
    frame.pts = index
    frame.time_base = Fraction(1, FPS)
    for packet in stream.encode(frame):
        container.mux(packet)


def previews(storyboard):
    PREVIEWS.mkdir(exist_ok=True)
    sheet = Image.new("RGB", (1920, math.ceil(len(storyboard["scenes"]) / 3) * 390), NAVY)
    drawing = ImageDraw.Draw(sheet)
    for index, scene in enumerate(storyboard["scenes"]):
        image = render_scene(scene, scene["duration"] * 0.62)
        assert max(ImageStat.Stat(image).var) > 200, f"Blank scene: {scene['id']}"
        image.save(PREVIEWS / f"{index + 1:02}_{scene['id']}.jpg", quality=95)
        sheet.paste(image.resize((640, 360), Image.Resampling.LANCZOS), ((index % 3) * 640, (index // 3) * 390))
        drawing.text(((index % 3) * 640 + 9, (index // 3) * 390 + 365), f"{index + 1:02} | {scene['start']:05.1f}s | {scene['id']}", font=font(18), fill=WHITE)
    sheet.save(PREVIEWS / "storyboard.jpg", quality=92)
    quote_scene = next(scene for scene in storyboard["scenes"] if scene["id"] == "quote")
    motion_frames = [np.asarray(render_scene(quote_scene, 1 + index / FPS).crop((200, 400, 1700, 660)), dtype=np.float32) for index in range(6)]
    frame_changes = [float(np.mean(np.abs(current - previous))) for previous, current in zip(motion_frames, motion_frames[1:])]
    assert min(frame_changes) > 0.001, frame_changes
    assert storyboard["scenes"][-1]["narration"] == "Thank you."
    probe = ROOT / "motion_probe_revised.mp4"
    with av.open(str(probe), "w") as container:
        stream = setup_stream(container)
        for index in range(60):
            scene = storyboard["scenes"][0] if index < 30 else quote_scene
            local_time = index / FPS if index < 30 else 1 + (index - 30) / FPS
            encode_frame(container, stream, render_scene(scene, local_time), index)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(probe)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 60
        assert frames[0].width == WIDTH and frames[0].height == HEIGHT
        assert np.mean(np.abs(frames[0].to_ndarray(format="rgb24").astype(float) - frames[-1].to_ndarray(format="rgb24"))) > 0.1
    print("All 30 scene layouts fit the 1920x1080 frame and are nonblank.")
    print("Motion encode/decode verified: 60 frames at 30 fps.")
    print(f"Continuous subpixel camera movement verified: minimum adjacent-frame change {min(frame_changes):.4f}.")
    print(f"Benefit highlights follow spoken sentence cues: {narration_cues('drivers', ('First', 'Second', 'Third', 'Fourth', 'Finally'))}")
    print(f"Storyboard preview: {PREVIEWS / 'storyboard.jpg'}")


def render_act(storyboard, act):
    SEGMENTS.mkdir(exist_ok=True)
    selected = [scene for scene in storyboard["scenes"] if scene["act"] == act]
    target = SEGMENTS / f"act_{act:02}.mp4"
    started = time.monotonic()
    frame_index = 0
    with av.open(str(target), "w", options={"movflags": "+faststart"}) as container:
        stream = setup_stream(container)
        for scene in selected:
            scene_position = storyboard["scenes"].index(scene)
            previous = storyboard["scenes"][scene_position - 1] if scene_position else None
            previous_frame = render_scene(previous, previous["duration"] - 0.01) if previous else Image.new("RGB", (WIDTH, HEIGHT), "black")
            count = round(scene["duration"] * FPS)
            for local_index in range(count):
                local_time = local_index / FPS
                image = render_scene(scene, local_time)
                transition_time = 0.4 if scene_position == 0 else 0.26
                if local_time < transition_time:
                    image = Image.blend(previous_frame, image, ease(local_time / transition_time))
                if scene["id"] == "close" and local_time > scene["duration"] - 1:
                    image = Image.blend(image, Image.new("RGB", image.size, "black"), ease(local_time - scene["duration"] + 1))
                encode_frame(container, stream, image, frame_index)
                frame_index += 1
            print(f"Act {act}: {scene['id']} rendered ({count} frames)", flush=True)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(target)) as container:
        video = container.streams.video[0]
        assert video.frames == frame_index, (video.frames, frame_index)
        assert abs(float(video.duration * video.time_base) - frame_index / FPS) < 0.01
    print(f"Verified act {act}: {frame_index / FPS:.2f}s, {frame_index} frames, rendered in {time.monotonic() - started:.1f}s.", flush=True)


def combine_segments(paths, target, duration):
    with av.open(str(paths[0])) as reference, av.open(str(target), "w", options={"movflags": "+faststart"}) as output:
        output.metadata.update({
            "title": "Lumen Zero-Touch NaaS Broker | Team ConnectIQ",
            "artist": "Team ConnectIQ",
            "comment": "Prototype demo using local mock APIs and illustrative pricing. AI-generated female narration and original ambient score.",
        })
        video_stream = output.add_stream_from_template(reference.streams.video[0])
        audio_stream = output.add_stream("aac", rate=SAMPLE_RATE)
        audio_stream.layout = "stereo"
        audio_stream.bit_rate = 192000
        audio_stream.time_base = Fraction(1, SAMPLE_RATE)

        def video_packets():
            frame_offset = 0
            for path in paths:
                with av.open(str(path)) as source:
                    stream = source.streams.video[0]
                    for packet in source.demux(stream):
                        if packet.dts is None:
                            continue
                        offset = int(Fraction(frame_offset, FPS) / packet.time_base)
                        packet.pts += offset
                        packet.dts += offset
                        packet.stream = video_stream
                        yield packet
                    frame_offset += stream.frames

        def audio_packets():
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SAMPLE_RATE)
            position = 0
            limit = round(duration * SAMPLE_RATE)
            with av.open(str(AUDIO / "final_mix.wav")) as source:
                for decoded in source.decode(audio=0):
                    for frame in resampler.resample(decoded):
                        remaining = limit - position
                        if remaining <= 0:
                            break
                        if frame.samples > remaining:
                            samples = np.ascontiguousarray(frame.to_ndarray()[:, :remaining])
                            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
                            frame.sample_rate = SAMPLE_RATE
                        frame.pts = position
                        frame.time_base = Fraction(1, SAMPLE_RATE)
                        position += frame.samples
                        for packet in audio_stream.encode(frame):
                            yield packet
                    if position >= limit:
                        break
            assert position == limit, (position, limit)
            for packet in audio_stream.encode():
                yield packet

        video_iterator = iter(video_packets())
        audio_iterator = iter(audio_packets())
        next_video = next(video_iterator, None)
        next_audio = next(audio_iterator, None)
        while next_video is not None or next_audio is not None:
            if next_audio is None or (
                next_video is not None
                and next_video.dts * next_video.time_base <= next_audio.dts * next_audio.time_base
            ):
                output.mux(next_video)
                next_video = next(video_iterator, None)
            else:
                output.mux(next_audio)
                next_audio = next(audio_iterator, None)
    print(f"Assembled: {target}", flush=True)


def assembly_probe():
    source = ROOT / "motion_probe_revised.mp4"
    target = ROOT / "assembly_probe_revised.mp4"
    combine_segments([source, source], target, 4.0)
    with av.open(str(target)) as container:
        assert 4.0 <= container.duration / av.time_base < 4.05
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.audio[0].codec_context.name == "aac"
        assert sum(1 for frame in container.decode(video=0)) == 120
    with av.open(str(target)) as container:
        peak = max(float(np.max(np.abs(frame.to_ndarray()))) for frame in container.decode(audio=0))
        assert 0.03 < peak < 1
    print("Assembly probe passed: 4 seconds, 120 video frames, stereo AAC, audible non-clipping narration.")


def verify_final(storyboard, target):
    expected_frames = round(storyboard["duration"] * FPS)
    samples = {
        round((scene["start"] + scene["duration"] * 0.62) * FPS): scene
        for scene in storyboard["scenes"]
    }
    scene_checks = []
    final_sheet = Image.new("RGB", (1920, 3900), NAVY)
    sheet_drawing = ImageDraw.Draw(final_sheet)
    with av.open(str(target)) as container:
        duration = container.duration / av.time_base
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        assert abs(duration - storyboard["duration"]) < 0.05 and duration <= 240.0, duration
        assert (video.width, video.height) == (WIDTH, HEIGHT)
        assert video.average_rate == FPS
        assert video.codec_context.name == "h264"
        assert audio.codec_context.name == "aac"
        assert audio.codec_context.sample_rate == SAMPLE_RATE
        assert len(audio.codec_context.layout.channels) == 2
        frame_count = 0
        for frame in container.decode(video=0):
            assert abs(frame.time - frame_count / FPS) < 0.002, (frame_count, frame.time)
            if frame_count in samples:
                scene = samples[frame_count]
                actual = frame.to_image()
                expected = render_scene(scene, frame.time - scene["start"])
                actual_small = np.asarray(actual.resize((320, 180)), dtype=float)
                expected_small = np.asarray(expected.resize((320, 180)), dtype=float)
                error = float(np.mean(np.abs(actual_small - expected_small)))
                assert error < 5, (scene["id"], error)
                assert float(np.std(actual_small)) > 12, scene["id"]
                scene_checks.append({"scene": scene["id"], "time": round(frame.time, 3), "mean_pixel_error": round(error, 3)})
                index = len(scene_checks) - 1
                final_sheet.paste(actual.resize((640, 360), Image.Resampling.LANCZOS), ((index % 3) * 640, (index // 3) * 390))
                sheet_drawing.text(((index % 3) * 640 + 9, (index // 3) * 390 + 365), f"{frame.time:06.2f}s | {scene['id']} | verified", font=font(18), fill=WHITE)
            frame_count += 1
    assert frame_count == expected_frames, (frame_count, expected_frames)
    assert len(scene_checks) == 30
    peak = 0.0
    square_sum = 0.0
    sample_count = 0
    with av.open(str(target)) as container:
        for frame in container.decode(audio=0):
            audio_samples = frame.to_ndarray()
            peak = max(peak, float(np.max(np.abs(audio_samples))))
            square_sum += float(np.sum(audio_samples.astype(np.float64) ** 2))
            sample_count += audio_samples.size
    rms = math.sqrt(square_sum / sample_count)
    assert 0.03 < peak < 1.0, peak
    assert rms > 0.015, rms
    final_sheet.save(PREVIEWS / "final_verified_frames.jpg", quality=92)
    report = {
        "file": target.name,
        "duration_seconds": duration,
        "video_frames": frame_count,
        "resolution": "1920x1080",
        "frames_per_second": FPS,
        "video_codec": "H.264 High Profile",
        "audio_codec": "AAC stereo / 48000 Hz / 192 kbps",
        "audio_peak_dbfs": round(20 * math.log10(peak), 2),
        "audio_rms_dbfs": round(20 * math.log10(rms), 2),
        "voice": storyboard["voice"],
        "voice_rate": storyboard["voice_rate"],
        "closing_narration": storyboard["scenes"][-1]["narration"],
        "ambient_gain": storyboard["music_gain"],
        "size_mb": round(target.stat().st_size / 1e6, 2),
        "all_frames_decoded": True,
        "all_audio_decoded": True,
        "scene_checks": scene_checks,
    }
    (ROOT / "verification_revised.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "scene_checks"}, indent=2), flush=True)
    print("All 30 scenes match their approved renders. All video frames and audio samples decode successfully.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preview", "act", "probe", "finish", "verify"])
    parser.add_argument("--act", type=int, choices=range(1, 7))
    arguments = parser.parse_args()
    storyboard = load_storyboard()
    if arguments.command == "preview":
        previews(storyboard)
    elif arguments.command == "act":
        assert arguments.act is not None
        render_act(storyboard, arguments.act)
    elif arguments.command == "probe":
        assembly_probe()
    else:
        target = ROOT / f"{storyboard['output_stem']}.mp4"
        if arguments.command == "finish":
            paths = [SEGMENTS / f"act_{act:02}.mp4" for act in range(1, 7)]
            assert all(path.exists() for path in paths), "An act is missing."
            combine_segments(paths, target, storyboard["duration"])
        verify_final(storyboard, target)


if __name__ == "__main__":
    main()