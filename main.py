from __future__ import annotations

import argparse
import csv
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
LINE_TRACKBAR_NAME = "Line Y"
LINE_START_TRACKBAR_NAME = "Line X1"
LINE_END_TRACKBAR_NAME = "Line X2"


@dataclass(slots=True)
class AppConfig:
    source: str | int
    model_path: str
    export_dir: Path
    line_start_x: int
    line_end_x: int
    line_y: int
    window_name: str
    vehicle_classes: tuple[int, ...]
    transport: str
    snapshot_interval: float
    reconnect_delay: float


def noop(_: int) -> None:
    return None


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
        help="Starting X coordinate of the counting line segment.",
    )
    parser.add_argument(
        "--line-end-x",
        type=int,
        default=1920,
        help="Ending X coordinate of the counting line segment.",
    )
    parser.add_argument(
        "--line-y",
        type=int,
        default=300,
        help="Y coordinate of the counting line.",
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
        line_start_x=args.line_start_x,
        line_end_x=args.line_end_x,
        line_y=args.line_y,
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


def count_vehicles(config: AppConfig) -> None:
    model = YOLO(config.model_path)
    capture = create_capture(config)
    exporter = HourlyCsvExporter(config.export_dir, config.source)
    counted_ids: set[int] = set()
    previous_center_y: dict[int, int] = {}
    count = 0
    last_export_path: Path | None = None
    line_start_x = config.line_start_x
    line_end_x = config.line_end_x
    line_y = config.line_y
    ui_initialized = False

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if not ui_initialized:
                max_x = max(1, frame.shape[1] - 1)
                line_y = max(0, min(line_y, frame.shape[0] - 1))
                line_start_x = max(0, min(line_start_x, max_x))
                line_end_x = max(0, min(line_end_x, max_x))
                line_start_x, line_end_x = sorted((line_start_x, line_end_x))
                cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
                cv2.createTrackbar(
                    LINE_TRACKBAR_NAME,
                    config.window_name,
                    line_y,
                    max(1, frame.shape[0] - 1),
                    noop,
                )
                cv2.createTrackbar(
                    LINE_START_TRACKBAR_NAME,
                    config.window_name,
                    line_start_x,
                    max_x,
                    noop,
                )
                cv2.createTrackbar(
                    LINE_END_TRACKBAR_NAME,
                    config.window_name,
                    line_end_x,
                    max_x,
                    noop,
                )
                ui_initialized = True

            line_y = cv2.getTrackbarPos(LINE_TRACKBAR_NAME, config.window_name)
            line_start_x = cv2.getTrackbarPos(LINE_START_TRACKBAR_NAME, config.window_name)
            line_end_x = cv2.getTrackbarPos(LINE_END_TRACKBAR_NAME, config.window_name)
            line_start_x, line_end_x = sorted((line_start_x, line_end_x))

            results = model.track(
                frame,
                persist=True,
                classes=list(config.vehicle_classes),
                verbose=False,
            )

            active_ids: set[int] = set()

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    track_id = get_track_id(box)
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    if track_id is not None:
                        active_ids.add(track_id)
                        previous_y = previous_center_y.get(track_id)
                        crossed_line = (
                            line_start_x <= center_x <= line_end_x
                            and previous_y is not None
                            and previous_y <= line_y < center_y
                            and track_id not in counted_ids
                        )
                        if crossed_line:
                            count += 1
                            counted_ids.add(track_id)
                            last_export_path = exporter.write_event(
                                timestamp=datetime.now(),
                                count=count,
                                track_id=track_id,
                                cls=cls,
                                center_x=center_x,
                                center_y=center_y,
                                line_start_x=line_start_x,
                                line_end_x=line_end_x,
                                line_y=line_y,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                            )
                        previous_center_y[track_id] = center_y

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

            previous_center_y = {
                track_id: previous_center_y[track_id]
                for track_id in active_ids
                if track_id in previous_center_y
            }

            draw_overlay(frame, line_start_x, line_end_x, line_y, count)
            cv2.imshow(config.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("w"), ord("W")):
                line_y = max(0, line_y - 5)
                cv2.setTrackbarPos(LINE_TRACKBAR_NAME, config.window_name, line_y)
            if key in (ord("s"), ord("S")):
                line_y = min(frame.shape[0] - 1, line_y + 5)
                cv2.setTrackbarPos(LINE_TRACKBAR_NAME, config.window_name, line_y)
    finally:
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


def draw_overlay(frame, line_start_x: int, line_end_x: int, line_y: int, count: int) -> None:
    cv2.line(frame, (line_start_x, line_y), (line_end_x, line_y), (0, 0, 255), 2)
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
        f"Line Y: {line_y}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Line X: {line_start_x} -> {line_end_x}",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Adjust Line Y / X1 / X2 sliders or press W/S, ESC to exit",
        (30, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


if __name__ == "__main__":
    count_vehicles(parse_args())
