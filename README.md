## Taiwan Traffic Camera Vehicle Counter

This project counts vehicles that cross a horizontal line in a video, webcam, or RTSP stream by using Ultralytics YOLO tracking.

It also supports camera web pages that expose a periodically refreshed snapshot image, such as some public Taiwan CCTV pages.

## 專案介紹

這個專案是一個即時車流計數工具，使用 Ultralytics YOLO 搭配 OpenCV，從道路監視器、RTSP、MJPEG 或快照型攝影機來源辨識車輛，並根據你設定的線段位置進行越線計數。

它的重點不是單純偵測畫面裡有幾台車，而是追蹤同一台車的移動軌跡，判斷車輛是否穿越指定的道路區段，避免同一台車被重複累計。

目前這個專案的示範場景，是針對關渡橋附近畫面中「台北往八里方向」的車道進行車流統計，也就是只統計穿越指定短線段、往八里關渡橋方向移動的車輛。

## 功能重點

- 支援預設公路局攝影機來源，也可以切換成 RTSP、MJPEG、snapshot 或 webcam
- 支援即時預覽畫面，可看到偵測框、車輛中心點、計數線段與目前統計數字
- 可在執行中調整短線段位置，只統計某一小段道路
- 支援 MJPEG 串流斷線重連，適合會週期性中斷的監視器來源
- 關閉視窗時會輸出最終統計數量

## 示意圖

![Vehicle Counter 示意圖](source/Snipaste_2026-03-31_11-39-40.png)

上圖可以看到程式在畫面中顯示車輛框線、追蹤 ID、車輛中心點，以及一條可調整長度的紅色計數線段。這個範例是設定在台北往八里關渡橋方向的目標車道上，只有當車輛中心點穿越這段線時，才會被納入統計。

## Demo 影片

<video src="source/demo.mp4" controls width="100%"></video>

如果你的 Markdown 檢視器不支援內嵌影片，可以直接開啟 [source/demo.mp4](source/demo.mp4)。

## 目前範例場景

- 攝影機來源：關渡橋附近公路局攝影機
- 統計方向：台北往八里關渡橋方向車道
- 統計方式：只計算穿越指定短線段的車輛

如果你想直接重現目前這個範例畫面，可以先用下面這組參數當起點：

```bash
uv run python main.py --line-start-x 140 --line-end-x 269 --line-y 126
```

實際值仍然可以依照當下畫面裁切、鏡頭角度或你要鎖定的車道位置，再用視窗中的 `Line Y`、`Line X1`、`Line X2` 滑桿微調。

## Requirements

- Windows, macOS, or Linux
- Python 3.13 or newer
- `uv`
- Network access for downloading Python packages and the YOLO model on first run

## Install uv

If you do not have `uv` yet, install it first.

### Windows PowerShell

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, open a new terminal and confirm:

```bash
uv --version
```

## Project Setup

Clone or open this folder, then move into the project directory:

```bash
cd baili
```

If you want `uv` to create and manage the local environment for this project, run:

```bash
uv sync
```

This command will:

- create `.venv` if it does not exist
- install dependencies from [pyproject.toml](pyproject.toml)
- make the environment ready for `uv run`

If you only want to install dependencies again after changes, run the same command:

```bash
uv sync
```

## Python Environment

You do not need to manually activate the virtual environment when using `uv run`.

Example:

```bash
uv run python main.py
```

If you still want to activate the environment manually:

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
source .venv/bin/activate
```

After manual activation, you can run:

```bash
python main.py
```

### Run

```bash
uv run python main.py
```

The default source is this camera:

```text
https://cctv-ss02.thb.gov.tw/T2-0K+060
```

You can still override it with another source:

```bash
uv run python main.py --source rtsp://username:password@camera-ip/stream --line-y 300
```

If you want to force live MJPEG stream mode:

```bash
uv run python main.py --transport stream
```

If you want to force snapshot mode:

```bash
uv run python main.py --transport snapshot --snapshot-interval 10
```

You can also pass a public camera page URL. The script will try to extract the underlying snapshot image automatically:

```bash
uv run python main.py --source "https://tw.live/cam/?id=CCTV-11-0020-000-004"
```

You can also use a webcam index:

```bash
uv run python main.py --source 0
```

Example with a short counting line segment:

```bash
uv run python main.py --line-start-x 700 --line-end-x 1100 --line-y 320
```

### Options

- `--model`: YOLO model path, default is `yolov8n.pt`
- `--line-start-x`: starting X coordinate of the counting line segment
- `--line-end-x`: ending X coordinate of the counting line segment
- `--line-y`: Y position of the counting line
- `--classes`: COCO class IDs to count, default is `2 3 5 7`
- `--transport`: `auto`, `stream`, or `snapshot`. Default is `auto`
- `--window-name`: OpenCV preview window title
- `--snapshot-interval`: polling interval for HTTP snapshot sources, default is `1.0`
- `--reconnect-delay`: reconnect delay for dropped MJPEG streams, default is `1.0`

### Preview Controls

- The preview window shows detection boxes, object center points, the counting line, and the current count.
- Drag the `Line Y`, `Line X1`, and `Line X2` sliders to move or shorten the counting line while the program is running.
- You can also press `W` and `S` to move the line up or down.
- Press `ESC` to close the window.
- When the window closes, the terminal prints the final count.

## How Counting Works

- The script uses YOLO object tracking to assign IDs to vehicles.
- A vehicle is counted once when its center point crosses from above the line to below the line.
- If you shorten the line using `Line X1` and `Line X2`, only vehicles crossing that road segment are counted.
- The final total is printed to the terminal when the window closes.

## First Run Behavior

- On first run, Ultralytics may download `yolov8n.pt` automatically.
- This file is the detection model weight and is required before detection starts.
- The first launch may therefore take longer than later launches.

## Common Commands

Install or update dependencies:

```bash
uv sync
```

Run with default camera:

```bash
uv run python main.py
```

Run with a custom source:

```bash
uv run python main.py --source "your-source"
```

Run with a different model:

```bash
uv run python main.py --model yolov8s.pt
```

## Troubleshooting

### `uv` command not found

- Restart the terminal after installing `uv`
- Check that `uv --version` works

### OpenCV window does not appear

- Make sure you are running on a desktop session, not a headless remote shell
- Try running from a normal PowerShell or terminal window

### Model download is slow or fails

- Check your internet connection
- Re-run the same command after network access is available

### Stream disconnects periodically

- This THB camera may close the MJPEG stream periodically
- The script will try to reconnect automatically in `auto` or `stream` mode
- You can tune reconnection behavior with `--reconnect-delay`

### Snapshot mode looks slow

- That is expected because snapshot mode fetches still images at intervals
- Increase or decrease `--snapshot-interval` depending on the source capability

### SSL certificate verification error

- Some THB camera endpoints use certificates that Python does not validate cleanly
- The script already includes a fallback for `thb.gov.tw` sources

### Notes

- The first run may download the YOLO model weights automatically.
- Counting is based on tracked object IDs crossing from above the line to below it.
- RTSP playback support depends on your local OpenCV and codec setup.
- For this THB camera, `auto` will prefer the live MJPEG stream and reconnect if the server closes the stream periodically.
- Snapshot-image sources are not true video streams, so tracking quality depends on the refresh rate and camera angle.
