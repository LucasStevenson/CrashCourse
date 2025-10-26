# CrashCourse - Driving Evaluation System

A real-time driving evaluation and coaching system that analyzes driving video footage to assess driver safety and behavior.

## Features

- Real-time object detection using YOLOv8
- Lane detection and departure warnings
- Time-to-collision (TTC) estimation
- Driving performance scoring with multiple dimensions:
  - Speeding violations
  - Lane keeping
  - Headway management
  - Smooth driving (harsh braking detection)
  - Traffic compliance (red lights, stop signs)
- Real-time coaching cues
- Multiple integration options (FastAPI, WebSocket, LiveKit)

## Project Structure

```
CrashCourse/
├── ai/src/               # AI inference engine
│   ├── api.py           # FastAPI endpoints
│   ├── detector.py      # YOLOv8 object detection
│   ├── rules.py         # Scoring and cuing logic
│   ├── lane_simple.py   # Lane detection
│   └── video_only.py    # Vision-based utilities
├── backend/             # WebSocket backend
│   └── app.py          # WebSocket server
└── livekit_backend/     # LiveKit integration
    └── livekit_backend.py
```

## Setup

### 1. Install Dependencies

```bash
pip install -r ai/requirements.txt
```

### 2. Download YOLOv8 Model

The YOLOv8n model will be automatically downloaded on first run, or you can place `yolov8n.pt` in the `ai/src/` directory.

### 3. Configure Environment (for LiveKit only)

If using LiveKit integration:

```bash
cd livekit_backend
cp .env.example .env
# Edit .env with your LiveKit credentials
```

## Usage

### Option 1: FastAPI Server (Recommended)

Start the inference API server:

```bash
cd ai/src
uvicorn api:app --host 0.0.0.0 --port 8000
```

API Endpoints:
- `POST /infer_frame` - Send frame + telemetry for inference
  - Parameters:
    - `image`: multipart image file
    - `telemetry`: JSON string with driving data
  - Returns: `{"cues": [...], "ttc": float, "detections": int}`

- `POST /end_session` - Get final driving score
  - Returns: `{"subscores": {...}, "final": float, "violations": {...}}`

### Option 2: WebSocket Server

Start the WebSocket server (automatically connects to FastAPI):

```bash
# Terminal 1: Start FastAPI server
cd ai/src
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2: Start WebSocket server
cd backend
python app.py
```

The WebSocket server listens on `ws://localhost:8765`

**Protocol:**
1. Send binary frame data (JPEG encoded)
2. Send JSON telemetry data
3. Receive real-time inference results
4. Send "DONE" message to get final score

### Option 3: LiveKit Integration

For Unity/WebRTC integration:

```bash
# Terminal 1: Start FastAPI server
cd ai/src
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2: Start LiveKit backend
cd livekit_backend
python livekit_backend.py
```

## Telemetry Data Format

```json
{
  "t": 1.5,                    // timestamp in seconds
  "speed_mps": 15.0,           // current speed in m/s
  "speed_limit_mps": 13.4,     // speed limit in m/s
  "throttle": 0.5,             // throttle position (0-1)
  "brake": 0.0,                // brake position (0-1)
  "steer_deg": -5.0,           // steering angle in degrees
  "lane_offset_m": 0.2,        // lane offset in meters (optional)
  "tl_state": "green",         // traffic light state (optional)
  "in_stop_zone": false,       // in stop zone (optional)
  "collision": false           // collision detected (boolean)
}
```

## Coaching Cues

The system generates the following real-time cues:
- `SLOW_DOWN` - Speed exceeds limit
- `KEEP_LANE` - Lane departure detected
- `INCREASE_HEADWAY` - Following too closely (low TTC)
- `SMOOTHER_BRAKE` - Harsh braking detected
- `BRAKE_NOW` - Red light/stop sign violation imminent

## Scoring

Final scores are calculated across 5 dimensions:
- **Speeding** (25% weight): Time spent over speed limit
- **Lane Keeping** (25% weight): Time spent out of lane
- **Headway** (20% weight): Time with inadequate TTC
- **Smoothness** (15% weight): Number of harsh braking events
- **Compliance** (15% weight): Red light violations and collisions

Each subscore ranges from 0-100, with the final score being a weighted average.

## Troubleshooting

**Model not found error:**
- Ensure `yolov8n.pt` is in `ai/src/` or let it auto-download
- Check internet connection for first-time model download

**Connection refused errors:**
- Verify FastAPI server is running on port 8000
- Check firewall settings

**No cues generated:**
- Verify telemetry data format matches specification
- Check that speed limits and thresholds are realistic

## Development

To run tests with sample video:

```bash
cd ai/src
python replay_test.py         # With synthetic telemetry
python replay_video_only.py   # Vision-only mode
```

## License

[Add your license here]
