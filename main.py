from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_VEHICLE_CLASSES = (2, 3, 5, 7)
DEFAULT_SOURCE = "https://cctv-ss02.thb.gov.tw/T2-0K+060"
DEFAULT_EXPORT_DIR = "exports"
DEFAULT_LINE_Y = 300
DEFAULT_LINE_BAND_HEIGHT = 30
DEFAULT_LEFT_MARGIN_RATIO = 0.15
DEFAULT_RIGHT_MARGIN_RATIO = 0.05
DEFAULT_TOP_MARGIN_RATIO = 0.10
DEFAULT_FRAME_CROP_LEFT_RATIO = 0.25
DEFAULT_FRAME_CROP_TOP_RATIO = 0.10
REGION_SETTINGS_FILE = Path("camera_region_settings.json")
LINE_TOP_TRACKBAR_NAME = "Line Y1"
LINE_BOTTOM_TRACKBAR_NAME = "Line Y2"
LINE_TOP_START_TRACKBAR_NAME = "Top X1"
LINE_TOP_END_TRACKBAR_NAME = "Top X2"
LINE_BOTTOM_START_TRACKBAR_NAME = "Bottom X1"
LINE_BOTTOM_END_TRACKBAR_NAME = "Bottom X2"


@dataclass(slots=True)
class AppConfig:
    source: str | int
    model_path: str
    export_dir: Path
    line_top_start_x: int
    line_top_end_x: int
    line_y: int
    line_bottom_start_x: int
    line_bottom_end_x: int
    line_y2: int
    window_name: str
    vehicle_classes: tuple[int, ...]
    transport: str
    snapshot_interval: float
    reconnect_delay: float


def noop(_: int) -> None:
    return None


def load_region_settings() -> dict[str, dict[str, int]]:
    if not REGION_SETTINGS_FILE.exists():
        return {}

    try:
        with REGION_SETTINGS_FILE.open("r", encoding="utf-8") as settings_file:
            data = json.load(settings_file)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    settings: dict[str, dict[str, int]] = {}
    for source_key, region in data.items():
        if not isinstance(source_key, str) or not isinstance(region, dict):
            continue
        legacy_fields = ("line_start_x", "line_end_x", "line_y1", "line_y2")
        current_fields = (
            "line_top_start_x",
            "line_top_end_x",
            "line_bottom_start_x",
            "line_bottom_end_x",
            "line_y1",
            "line_y2",
        )
        if all(isinstance(region.get(field), int) for field in current_fields):
            settings[source_key] = {
                "line_top_start_x": int(region["line_top_start_x"]),
                "line_top_end_x": int(region["line_top_end_x"]),
                "line_bottom_start_x": int(region["line_bottom_start_x"]),
                "line_bottom_end_x": int(region["line_bottom_end_x"]),
                "line_y1": int(region["line_y1"]),
                "line_y2": int(region["line_y2"]),
            }
        elif all(isinstance(region.get(field), int) for field in legacy_fields):
            settings[source_key] = {
                "line_top_start_x": int(region["line_start_x"]),
                "line_top_end_x": int(region["line_end_x"]),
                "line_bottom_start_x": int(region["line_start_x"]),
                "line_bottom_end_x": int(region["line_end_x"]),
                "line_y1": int(region["line_y1"]),
                "line_y2": int(region["line_y2"]),
            }
    return settings


def load_region_setting(source: str | int) -> dict[str, int] | None:
    return load_region_settings().get(str(source))


def save_region_setting(
    source: str | int,
    line_top_start_x: int,
    line_top_end_x: int,
    line_bottom_start_x: int,
    line_bottom_end_x: int,
    line_y1: int,
    line_y2: int,
) -> None:
    settings = load_region_settings()
    settings[str(source)] = {
        "line_top_start_x": int(line_top_start_x),
        "line_top_end_x": int(line_top_end_x),
        "line_bottom_start_x": int(line_bottom_start_x),
        "line_bottom_end_x": int(line_bottom_end_x),
        "line_y1": int(line_y1),
        "line_y2": int(line_y2),
    }

    with REGION_SETTINGS_FILE.open("w", encoding="utf-8") as settings_file:
        json.dump(settings, settings_file, ensure_ascii=True, indent=2, sort_keys=True)


def get_default_line_x_bounds(frame_width: int) -> tuple[int, int]:
    max_x = max(1, frame_width - 1)
    line_start_x = max(0, min(max_x, int(frame_width * DEFAULT_LEFT_MARGIN_RATIO)))
    line_end_x = max(line_start_x, min(max_x, int(frame_width * (1.0 - DEFAULT_RIGHT_MARGIN_RATIO))))
    return line_start_x, line_end_x


def get_default_line_y_bounds(frame_height: int, band_height: int) -> tuple[int, int]:
    max_y = max(1, frame_height - 1)
    line_y1 = max(0, min(max_y, int(frame_height * DEFAULT_TOP_MARGIN_RATIO)))
    line_y2 = min(max_y, line_y1 + max(1, band_height))
    return line_y1, max(line_y1, line_y2)


def crop_frame(frame: np.ndarray) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    crop_left = min(frame_width - 1, max(0, int(frame_width * DEFAULT_FRAME_CROP_LEFT_RATIO)))
    crop_top = min(frame_height - 1, max(0, int(frame_height * DEFAULT_FRAME_CROP_TOP_RATIO)))
    cropped = frame[crop_top:, crop_left:]
    if cropped.size == 0:
        return frame
    return cropped


class HourlyCsvExporter:
    def __init__(self, export_dir: Path, source: str | int) -> None:
        self.export_dir = export_dir
        self.source = str(source)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def write_event(
        self,
        *,
        timestamp: datetime,
        count: int,
        track_id: int,
        cls: int,
        center_x: int,
        center_y: int,
        line_start_x: int,
        line_end_x: int,
        line_y: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Path:
        file_path = self.export_dir / f"vehicle_counts_{timestamp.strftime('%Y%m%d_%H00')}.csv"
        file_exists = file_path.exists()

        with file_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "hour_bucket",
                        "source",
                        "count",
                        "track_id",
                        "class_id",
                        "center_x",
                        "center_y",
                        "line_start_x",
                        "line_end_x",
                        "line_y",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    ]
                )

            writer.writerow(
                [
                    timestamp.isoformat(timespec="seconds"),
                    timestamp.strftime("%Y-%m-%d %H:00:00"),
                    self.source,
                    count,
                    track_id,
                    cls,
                    center_x,
                    center_y,
                    line_start_x,
                    line_end_x,
                    line_y,
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

        return file_path


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Count vehicles crossing a horizontal line in a video or RTSP stream."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Video source. Use an RTSP URL, webcam index, or a local video file path.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO model path.",
    )
    parser.add_argument(
        "--export-dir",
        default=DEFAULT_EXPORT_DIR,
        help="Directory where hourly CSV files are written.",
    )
    parser.add_argument(
        "--line-start-x",
        type=int,
        default=0,
        help="Starting X coordinate of the first counting line segment.",
    )
    parser.add_argument(
        "--line-end-x",
        type=int,
        default=1920,
        help="Ending X coordinate of the first counting line segment.",
    )
    parser.add_argument(
        "--line2-start-x",
        type=int,
        default=None,
        help="Starting X coordinate of the second counting line segment. Defaults to --line-start-x.",
    )
    parser.add_argument(
        "--line2-end-x",
        type=int,
        default=None,
        help="Ending X coordinate of the second counting line segment. Defaults to --line-end-x.",
    )
    parser.add_argument(
        "--line-y",
        type=int,
        default=DEFAULT_LINE_Y,
        help="Y coordinate of the first counting line.",
    )
    parser.add_argument(
        "--line-y2",
        type=int,
        default=None,
        help=f"Y coordinate of the second counting line. Defaults to --line-y + {DEFAULT_LINE_BAND_HEIGHT}.",
    )
    parser.add_argument(
        "--window-name",
        default="Vehicle Counter",
        help="OpenCV display window name.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(DEFAULT_VEHICLE_CLASSES),
        help="COCO class IDs to track. Defaults to car, motorcycle, bus, truck.",
    )
    parser.add_argument(
        "--transport",
        choices=("auto", "stream", "snapshot"),
        default="auto",
        help="HTTP source handling mode. 'auto' prefers live MJPEG streams when available.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds when the source is an HTTP snapshot image.",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=1.0,
        help="Delay in seconds before reconnecting a dropped MJPEG stream.",
    )
    args = parser.parse_args()

    return AppConfig(
        source=normalize_source(args.source),
        model_path=args.model,
        export_dir=Path(args.export_dir),
        line_top_start_x=args.line_start_x,
        line_top_end_x=args.line_end_x,
        line_y=args.line_y,
        line_bottom_start_x=args.line2_start_x if args.line2_start_x is not None else args.line_start_x,
        line_bottom_end_x=args.line2_end_x if args.line2_end_x is not None else args.line_end_x,
        line_y2=args.line_y2 if args.line_y2 is not None else args.line_y + DEFAULT_LINE_BAND_HEIGHT,
        window_name=args.window_name,
        vehicle_classes=tuple(args.classes),
        transport=args.transport,
        snapshot_interval=args.snapshot_interval,
        reconnect_delay=args.reconnect_delay,
    )


def normalize_source(raw_source: str) -> str | int:
    if raw_source.isdigit():
        return int(raw_source)
    return raw_source


def open_capture(source: str | int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return capture


class SnapshotCapture:
    def __init__(self, snapshot_url: str, poll_interval: float) -> None:
        self.snapshot_url = snapshot_url
        self.poll_interval = max(0.1, poll_interval)
        self._closed = False
        self._last_frame_time = 0.0

    def isOpened(self) -> bool:
        return not self._closed

    def read(self) -> tuple[bool, np.ndarray | None]:
        now = time.monotonic()
        wait_time = self.poll_interval - (now - self._last_frame_time)
        if wait_time > 0:
            time.sleep(wait_time)

        try:
            frame = fetch_snapshot_frame(self.snapshot_url)
        except RuntimeError:
            return False, None

        self._last_frame_time = time.monotonic()
        return True, frame

    def release(self) -> None:
        self._closed = True


class MjpegCapture:
    def __init__(self, stream_url: str, reconnect_delay: float) -> None:
        self.stream_url = stream_url
        self.reconnect_delay = max(0.1, reconnect_delay)
        self._closed = False
        self._response = None
        self._buffer = bytearray()

    def isOpened(self) -> bool:
        return not self._closed

    def read(self) -> tuple[bool, np.ndarray | None]:
        while not self._closed:
            if self._response is None:
                try:
                    self._response = open_mjpeg_stream(self.stream_url)
                except RuntimeError:
                    time.sleep(self.reconnect_delay)
                    continue

            try:
                chunk = self._response.read(4096)
                if not chunk:
                    raise RuntimeError("MJPEG stream ended")
                self._buffer.extend(chunk)
                frame = self._extract_frame()
                if frame is not None:
                    return True, frame
            except Exception:
                self._disconnect()
                if not self._closed:
                    time.sleep(self.reconnect_delay)

        return False, None

    def _extract_frame(self) -> np.ndarray | None:
        start = self._buffer.find(b"\xff\xd8")
        if start == -1:
            self._trim_buffer()
            return None

        end = self._buffer.find(b"\xff\xd9", start + 2)
        if end == -1:
            if start > 0:
                del self._buffer[:start]
            self._trim_buffer()
            return None

        jpeg_bytes = bytes(self._buffer[start : end + 2])
        del self._buffer[: end + 2]

        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

    def _trim_buffer(self) -> None:
        max_buffer_size = 2_000_000
        if len(self._buffer) > max_buffer_size:
            del self._buffer[:-max_buffer_size]

    def _disconnect(self) -> None:
        if self._response is not None:
            with suppress(Exception):
                self._response.close()
        self._response = None
        self._buffer.clear()

    def release(self) -> None:
        self._closed = True
        self._disconnect()


def create_capture(config: AppConfig) -> cv2.VideoCapture | SnapshotCapture | MjpegCapture:
    source = config.source
    if isinstance(source, int):
        return open_capture(source)

    if source.startswith(("http://", "https://")):
        resolved_source = resolve_http_source(source)
        snapshot_candidate = get_snapshot_candidate(resolved_source)
        media_type = probe_http_media_type(resolved_source)

        if config.transport == "stream":
            if media_type.startswith("multipart/") or media_type.startswith("video/"):
                return MjpegCapture(resolved_source, config.reconnect_delay)
            raise RuntimeError(f"Live stream is not available for source: {resolved_source}")

        if config.transport == "snapshot" and snapshot_candidate is not None:
            snapshot_media_type = probe_http_media_type(snapshot_candidate)
            if snapshot_media_type.startswith("image/"):
                return SnapshotCapture(snapshot_candidate, config.snapshot_interval)
            raise RuntimeError(f"Snapshot is not available for source: {resolved_source}")

        if media_type.startswith("multipart/") or media_type.startswith("video/"):
            return MjpegCapture(resolved_source, config.reconnect_delay)

        if snapshot_candidate is not None:
            snapshot_media_type = probe_http_media_type(snapshot_candidate)
            if snapshot_media_type.startswith("image/"):
                return SnapshotCapture(snapshot_candidate, config.snapshot_interval)

        if media_type.startswith("image/"):
            return SnapshotCapture(resolved_source, config.snapshot_interval)

        if resolved_source != source:
            try:
                return open_capture(resolved_source)
            except RuntimeError:
                if is_snapshot_url(resolved_source):
                    return SnapshotCapture(resolved_source, config.snapshot_interval)

    return open_capture(source)


def resolve_http_source(source: str) -> str:
    if is_snapshot_url(source):
        return source

    if not should_resolve_as_html_page(source):
        return source

    html = fetch_text(source)
    snapshot_url = extract_snapshot_url(html, source)
    if snapshot_url is None:
        return source
    return snapshot_url


def is_snapshot_url(source: str) -> bool:
    normalized = source.lower()
    return "/snapshot" in normalized or normalized.endswith((".jpg", ".jpeg", ".png"))


def get_snapshot_candidate(source: str) -> str | None:
    if is_snapshot_url(source):
        return source

    if source.startswith(("http://", "https://")):
        return source.rstrip("/") + "/snapshot"

    return None


def should_resolve_as_html_page(source: str) -> bool:
    parsed = urlparse(source)
    host = parsed.netloc.lower()

    if host.endswith("tw.live"):
        return True

    if parsed.path.endswith((".html", "/")):
        return True

    return False


def should_allow_insecure_ssl(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host.endswith("thb.gov.tw")


def open_url(request: Request, timeout: int = 15):
    try:
        return urlopen(request, timeout=timeout)
    except URLError as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, ssl.SSLCertVerificationError) and should_allow_insecure_ssl(request.full_url):
            insecure_context = ssl._create_unverified_context()
            return urlopen(request, timeout=timeout, context=insecure_context)
        raise


def open_mjpeg_stream(url: str):
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "multipart/x-mixed-replace,image/jpeg,*/*;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        },
    )
    try:
        return open_url(request, timeout=30)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Unable to open MJPEG stream: {url}") from exc


def fetch_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with open_url(request, timeout=15) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Unable to fetch URL: {url}") from exc


def probe_http_media_type(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    try:
        with open_url(request, timeout=15) as response:
            return response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    except (HTTPError, URLError):
        return ""


def extract_snapshot_url(html: str, base_url: str) -> str | None:
    primary_media_match = re.search(
        r'<div class="image-container">.*?<img[^>]+data-src="([^"]+)"',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if primary_media_match:
        return urljoin(base_url, primary_media_match.group(1))

    patterns = (
        r'data-src="([^"]+?/snapshot(?:\?[^"]*)?)"',
        r'src="([^"]+?/snapshot(?:\?[^"]*)?)"',
        r'content="([^"]+?/snapshot(?:\?[^"]*)?)"',
        r'data-src="([^"]+\.(?:jpg|jpeg|png)(?:\?[^"]*)?)"',
    )
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return urljoin(base_url, match.group(1))
    return None


def fetch_snapshot_frame(snapshot_url: str) -> np.ndarray:
    request = Request(
        snapshot_url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    try:
        with open_url(request, timeout=15) as response:
            image_bytes = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Unable to fetch snapshot: {snapshot_url}") from exc

    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Unable to decode snapshot image: {snapshot_url}")
    return frame


def apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    lookup = np.array([((index / 255.0) ** gamma) * 255 for index in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lookup)


def enhance_low_light_frame(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[:, :, 2].mean())
    if brightness >= 90:
        return frame

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    gamma = 0.75 if brightness < 55 else 0.85
    return apply_gamma(enhanced, gamma)


def build_untracked_detection_key(
    cls: int, center_x: int, center_y: int, width: int, height: int
) -> tuple[int, int, int, int, int]:
    return (
        cls,
        center_x // 20,
        center_y // 20,
        max(1, width // 20),
        max(1, height // 20),
    )


def is_recent_count_match(
    recent_event: tuple[int, int, int, int, int, int],
    cls: int,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    frame_count: int,
) -> bool:
    recent_cls, recent_center_x, recent_center_y, recent_width, recent_height, recent_frame = recent_event
    if recent_cls != cls:
        return False
    if frame_count - recent_frame > 90:
        return False

    distance_threshold = max(40, min(max(width, height), 120))
    if abs(center_x - recent_center_x) > distance_threshold:
        return False
    if abs(center_y - recent_center_y) > distance_threshold:
        return False

    width_ratio = width / max(1, recent_width)
    height_ratio = height / max(1, recent_height)
    return 0.5 <= width_ratio <= 2.0 and 0.5 <= height_ratio <= 2.0


def count_vehicles(config: AppConfig) -> None:
    # model = YOLO(config.model_path)
    model = YOLO(config.model_path)
    model.to("cuda")       # 強制用GPU
    model.fuse()           # 加速
    model.model.half()     # FP16（超重要）
    capture = create_capture(config)
    exporter = HourlyCsvExporter(config.export_dir, config.source)
    counted_ids: set[int] = set()
    counted_untracked: dict[tuple[int, int, int, int, int], int] = {}
    recent_counted_events: list[tuple[int, int, int, int, int, int]] = []
    count = 0
    last_export_path: Path | None = None
    line_top_start_x = config.line_top_start_x
    line_top_end_x = config.line_top_end_x
    line_bottom_start_x = config.line_bottom_start_x
    line_bottom_end_x = config.line_bottom_end_x
    line_y1, line_y2 = sorted((config.line_y, config.line_y2))
    saved_region = load_region_setting(config.source)
    ui_initialized = False
    frame_count = 0
    last_results = None
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame = crop_frame(frame)
            frame = cv2.resize(frame, (416, 416))  # 降解析度（超重要）
            frame = enhance_low_light_frame(frame)

            # # ROI（只抓下半部道路）
            # h = frame.shape[0]
            # frame = frame[int(h * 0.4) : h, :]
            frame_count += 1

            if not ui_initialized:
                max_x = max(1, frame.shape[1] - 1)
                max_y = max(1, frame.shape[0] - 1)
                if saved_region is not None:
                    line_top_start_x = saved_region["line_top_start_x"]
                    line_top_end_x = saved_region["line_top_end_x"]
                    line_bottom_start_x = saved_region["line_bottom_start_x"]
                    line_bottom_end_x = saved_region["line_bottom_end_x"]
                    line_y1, line_y2 = sorted((saved_region["line_y1"], saved_region["line_y2"]))
                elif line_top_start_x == 0 and line_top_end_x == 1920:
                    line_top_start_x, line_top_end_x = get_default_line_x_bounds(frame.shape[1])
                    line_bottom_start_x, line_bottom_end_x = line_top_start_x, line_top_end_x
                    if config.line_y == DEFAULT_LINE_Y and config.line_y2 == DEFAULT_LINE_Y + DEFAULT_LINE_BAND_HEIGHT:
                        line_y1, line_y2 = get_default_line_y_bounds(
                            frame.shape[0],
                            DEFAULT_LINE_BAND_HEIGHT,
                        )
                line_y1 = max(0, min(line_y1, max_y))
                line_y2 = max(0, min(line_y2, max_y))
                line_y1, line_y2 = sorted((line_y1, line_y2))
                line_top_start_x = max(0, min(line_top_start_x, max_x))
                line_top_end_x = max(0, min(line_top_end_x, max_x))
                line_bottom_start_x = max(0, min(line_bottom_start_x, max_x))
                line_bottom_end_x = max(0, min(line_bottom_end_x, max_x))
                line_top_start_x, line_top_end_x = sorted((line_top_start_x, line_top_end_x))
                line_bottom_start_x, line_bottom_end_x = sorted((line_bottom_start_x, line_bottom_end_x))
                cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
                cv2.createTrackbar(
                    LINE_TOP_TRACKBAR_NAME,
                    config.window_name,
                    line_y1,
                    max_y,
                    noop,
                )
                cv2.createTrackbar(
                    LINE_BOTTOM_TRACKBAR_NAME,
                    config.window_name,
                    line_y2,
                    max_y,
                    noop,
                )
                cv2.createTrackbar(
                    LINE_TOP_START_TRACKBAR_NAME,
                    config.window_name,
                    line_top_start_x,
                    max_x,
                    noop,
                )
                cv2.createTrackbar(
                    LINE_TOP_END_TRACKBAR_NAME,
                    config.window_name,
                    line_top_end_x,
                    max_x,
                    noop,
                )
                cv2.createTrackbar(
                    LINE_BOTTOM_START_TRACKBAR_NAME,
                    config.window_name,
                    line_bottom_start_x,
                    max_x,
                    noop,
                )
                cv2.createTrackbar(
                    LINE_BOTTOM_END_TRACKBAR_NAME,
                    config.window_name,
                    line_bottom_end_x,
                    max_x,
                    noop,
                )
                ui_initialized = True

            line_y1 = cv2.getTrackbarPos(LINE_TOP_TRACKBAR_NAME, config.window_name)
            line_y2 = cv2.getTrackbarPos(LINE_BOTTOM_TRACKBAR_NAME, config.window_name)
            line_y1, line_y2 = sorted((line_y1, line_y2))
            line_top_start_x = cv2.getTrackbarPos(LINE_TOP_START_TRACKBAR_NAME, config.window_name)
            line_top_end_x = cv2.getTrackbarPos(LINE_TOP_END_TRACKBAR_NAME, config.window_name)
            line_bottom_start_x = cv2.getTrackbarPos(LINE_BOTTOM_START_TRACKBAR_NAME, config.window_name)
            line_bottom_end_x = cv2.getTrackbarPos(LINE_BOTTOM_END_TRACKBAR_NAME, config.window_name)
            line_top_start_x, line_top_end_x = sorted((line_top_start_x, line_top_end_x))
            line_bottom_start_x, line_bottom_end_x = sorted((line_bottom_start_x, line_bottom_end_x))
            counting_band = build_counting_band_polygon(
                line_top_start_x,
                line_top_end_x,
                line_y1,
                line_bottom_start_x,
                line_bottom_end_x,
                line_y2,
            )

            if frame_count % 5 == 0:
                results = model.track(
                    frame,
                    persist=True,
                    classes=list(config.vehicle_classes),
                    verbose=False,
                    device=0,
                    imgsz=416,
                    half=True,
                )
                last_results = results
            else:
                results = last_results if last_results is not None else []

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    track_id = get_track_id(box)
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_width = max(1, x2 - x1)
                    box_height = max(1, y2 - y1)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    inside_region = point_in_polygon((center_x, center_y), counting_band)
                    matches_recent_count = any(
                        is_recent_count_match(
                            recent_event,
                            cls,
                            center_x,
                            center_y,
                            box_width,
                            box_height,
                            frame_count,
                        )
                        for recent_event in recent_counted_events
                    )

                    if (
                        inside_region
                        and not matches_recent_count
                        and track_id is not None
                        and track_id not in counted_ids
                    ):
                        count += 1
                        counted_ids.add(track_id)
                        recent_counted_events.append(
                            (cls, center_x, center_y, box_width, box_height, frame_count)
                        )
                        last_export_path = exporter.write_event(
                            timestamp=datetime.now(),
                            count=count,
                            track_id=track_id,
                            cls=cls,
                            center_x=center_x,
                            center_y=center_y,
                            line_start_x=line_top_start_x,
                            line_end_x=line_top_end_x,
                            line_y=(line_y1 + line_y2) // 2,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    elif inside_region and not matches_recent_count and track_id is None:
                        detection_key = build_untracked_detection_key(
                            cls, center_x, center_y, box_width, box_height
                        )
                        last_counted_frame = counted_untracked.get(detection_key)
                        if last_counted_frame is None or frame_count - last_counted_frame > 30:
                            count += 1
                            counted_untracked[detection_key] = frame_count
                            recent_counted_events.append(
                                (cls, center_x, center_y, box_width, box_height, frame_count)
                            )
                            last_export_path = exporter.write_event(
                                timestamp=datetime.now(),
                                count=count,
                                track_id=-1,
                                cls=cls,
                                center_x=center_x,
                                center_y=center_y,
                                line_start_x=line_top_start_x,
                                line_end_x=line_top_end_x,
                                line_y=(line_y1 + line_y2) // 2,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                            )

                    counted_untracked = {
                        key: last_seen_frame
                        for key, last_seen_frame in counted_untracked.items()
                        if frame_count - last_seen_frame <= 120
                    }
                    recent_counted_events = [
                        recent_event
                        for recent_event in recent_counted_events
                        if frame_count - recent_event[5] <= 120
                    ]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 4, (255, 255, 0), -1)
                    label = format_label(box, track_id)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            draw_overlay(
                frame,
                line_top_start_x,
                line_top_end_x,
                line_y1,
                line_bottom_start_x,
                line_bottom_end_x,
                line_y2,
                count,
            )
            cv2.imshow(config.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("w"), ord("W")):
                line_y1, line_y2 = shift_counting_band(line_y1, line_y2, -5, frame.shape[0] - 1)
                cv2.setTrackbarPos(LINE_TOP_TRACKBAR_NAME, config.window_name, line_y1)
                cv2.setTrackbarPos(LINE_BOTTOM_TRACKBAR_NAME, config.window_name, line_y2)
            if key in (ord("s"), ord("S")):
                line_y1, line_y2 = shift_counting_band(line_y1, line_y2, 5, frame.shape[0] - 1)
                cv2.setTrackbarPos(LINE_TOP_TRACKBAR_NAME, config.window_name, line_y1)
                cv2.setTrackbarPos(LINE_BOTTOM_TRACKBAR_NAME, config.window_name, line_y2)
    finally:
        if ui_initialized:
            with suppress(Exception):
                save_region_setting(
                    config.source,
                    line_top_start_x,
                    line_top_end_x,
                    line_bottom_start_x,
                    line_bottom_end_x,
                    line_y1,
                    line_y2,
                )
        with suppress(Exception):
            capture.release()
        cv2.destroyAllWindows()
        print(f"Final count: {count}")
        if last_export_path is not None:
            print(f"CSV export: {last_export_path}")
        else:
            print("CSV export: no crossings recorded yet")


def get_track_id(box) -> int | None:
    if box.id is None:
        return None
    return int(box.id[0])


def format_label(box, track_id: int | None) -> str:
    parts = [f"cls {int(box.cls[0])}"]
    if track_id is not None:
        parts.append(f"id {track_id}")
    return " | ".join(parts)


def build_counting_band_polygon(
    line_top_start_x: int,
    line_top_end_x: int,
    line_y1: int,
    line_bottom_start_x: int,
    line_bottom_end_x: int,
    line_y2: int,
) -> np.ndarray:
    return np.array(
        [
            (line_top_start_x, line_y1),
            (line_top_end_x, line_y1),
            (line_bottom_end_x, line_y2),
            (line_bottom_start_x, line_y2),
        ],
        dtype=np.int32,
    )


def point_in_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


def shift_counting_band(line_y1: int, line_y2: int, delta: int, max_y: int) -> tuple[int, int]:
    band_height = line_y2 - line_y1
    new_line_y1 = max(0, min(line_y1 + delta, max_y - band_height))
    return new_line_y1, new_line_y1 + band_height


def draw_overlay(
    frame,
    line_top_start_x: int,
    line_top_end_x: int,
    line_y1: int,
    line_bottom_start_x: int,
    line_bottom_end_x: int,
    line_y2: int,
    count: int,
) -> None:
    overlay = frame.copy()
    band_polygon = build_counting_band_polygon(
        line_top_start_x,
        line_top_end_x,
        line_y1,
        line_bottom_start_x,
        line_bottom_end_x,
        line_y2,
    )
    cv2.fillConvexPoly(overlay, band_polygon, (0, 64, 255))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.line(frame, (line_top_start_x, line_y1), (line_top_end_x, line_y1), (0, 0, 255), 2)
    cv2.line(frame, (line_bottom_start_x, line_y2), (line_bottom_end_x, line_y2), (0, 165, 255), 2)
    cv2.putText(
        frame,
        f"Count: {count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Band Y: {line_y1} -> {line_y2}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Top X: {line_top_start_x} -> {line_top_end_x}",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Bottom X: {line_bottom_start_x} -> {line_bottom_end_x}",
        (30, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Adjust Y1 / Y2 / Top X / Bottom X sliders or press W/S to move band, ESC to exit",
        (30, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


if __name__ == "__main__":
    count_vehicles(parse_args())
