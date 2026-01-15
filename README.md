# Interview Monitoring System

A robust, real-time web application designed to monitor candidate integrity during online interviews. This system detects suspicious behaviors such as multiple faces, face absence, prolonged silence, and window switching using a combination of Computer Vision, Audio Signal Processing, and Heuristics.

## 📋 Table of Contents
- [Overview](#overview)
- [Deep Dive: Architecture](#deep-dive-architecture)
- [Deep Dive: Workflow & Logic](#deep-dive-workflow--logic)
- [Data & State Management](#data--state-management)
- [Machine Learning & Algorithms](#machine-learning--algorithms)
- [Setup & Installation](#setup--installation)
- [Scoring System](#scoring-system)


## 🔍 Overview
The **Interview Monitoring System** provides an automated proctoring visual and audio environment. It captures video and audio streams via the browser, analyzes them in real-time on a Django backend, and logs specific violations. The system generates a final integrity score and risk assessment (Green/Yellow/Red) at the end of the session.

## 🏗 Deep Dive: Architecture

The system operates on a **Monolithic MVC Pattern** (Model-View-Controller) powered by Django, with a "Thick Client" (JS) for data acquisition and "Thick Server" (Python) for analysis.

### 1. The Frontend Layer (Data Acquisition)
The client-side is responsible for accessing hardware sensors and serializing raw data.
*   **Video Pipeline**:
    *   The `msg stream` from the Webcam is rendered to a hidden `<video>` element.
    *   A `<canvas>` context draws the current video frame every **1000ms (1 FPS)**.
    *   `canvas.toDataURL('image/jpeg', 0.5)` converts this frame into a reduced-quality **Base64 string** to optimize network bandwidth (reducing 1080p raw frames to manageable ~20-50KB payloads).
*   **Audio Pipeline**:
    *   Uses the **Web Audio API**. An `AudioContext` creates a `MediaStreamSource` from the microphone.
    *   An `AnalyserNode` with `fftSize=1024` performs a Fast Fourier Transform.
    *   A `ScriptProcessorNode` (size 2048) collects frequency mapping data and calculates the **Average Amplitude**.
    *   *Optimization*: Instead of streaming raw audio (heavy bandwidth), the **silence/activity decision is made locally in JS**. Only a boolean flag (`is_silent`) is sent to the server.

### 2. The Backend Layer (Analysis & State)
The Django server acts as the central brain.
*   **Request Handling**:
    *   REST-like endpoints (`/api/process_frame/`, `/api/send_audio_activity/`) accept JSON payloads.
    *   `@csrf_exempt` is used for these specific API endpoints to facilitate easier async fetches (though in production, CSRF tokens should be passed in headers).
*   **Image Processing**:
    *   The Base64 string is cleaned (header removal) and decoded into bytes.
    *   `numpy.frombuffer` allows OpenCV (`cv2`) to interpret the bytes as an image array (Matrix).
*   **Violations Engine**:
    *   The engine is **Stateless** per request but **Stateful** via the Database.
    *   It compares the current analysis result (e.g., "No Face") against the `last_face_seen` timestamp stored in the Database `InterviewSession`.

### 3. Database Schema
*   **Candidate**: Static identity (Name, Email).
*   **InterviewSession**: The central state machine.
    *   `last_face_seen` (DateTime): Updated on every valid frame. Used to calculate "Face Missing Duration".
    *   `last_audio_activity` (DateTime): Updated on every non-silent signal. Used to calculate "Silence Duration".
    *   `risk_level` (Enum): Final persistence of the exam result.
*   **ViolationLog**: A transaction log of every detected bad event. One-to-Many relationship with Session.

## 🔄 Deep Dive: Workflow & Logic

### Phase 1: Session Handshake
1.  **User Entry**: User inputs Name/Email on `index.html`.
2.  **Session Creation**:
    *   Backend checks if `Candidate` exists (Get or Create).
    *   Backend creates a new `InterviewSession` with `is_active=True`.
    *   **CRITICAL**: The `session_id` is returned to the client and stored in `sessionStorage`. This token is required for all subsequent API calls to link data streams to the correct user.

### Phase 2: The Monitoring Loop (Pulse)
The system runs two parallel asynchronous loops on the client side:

#### A. The Visual Loop (1000ms Interval)
1.  **Capture**: JS takes a snapshot.
2.  **Send**: POST to `/api/process_frame/`.
3.  **Analyze (Server)**:
    *   `FaceAnalyzer` detects faces.
    *   **Logic Branch**:
        *   *If 0 Faces*: Check `(Now - session.last_face_seen)`.
            *   If > `FACE_ABSENCE_THRESHOLD` (5s), check Debounce.
            *   **Debounce**: Have we logged this violation in the last 2 seconds? If no, **Write to DB**.
        *   *If > 1 Face*: Immediate "Multiple Faces" violation logged.
        *   *If 1 Face*: Update `session.last_face_seen = Now`.
4.  **Feedback**: Server responds with `{ face_status: 'no_face', violation: 'FACE_MISSING' }`.
5.  **UI Update**: JS logic receives this. If `violation` is present, it triggers a "Toast" alert and turns the status indicator Red.

#### B. The Audio Loop (1000ms Interval)
1.  **Check**: JS checks the averaged volume against a threshold (5).
2.  **Send**: POST to `/api/send_audio_activity/` with `{ is_silent: true/false }`.
3.  **Analyze (Server)**:
    *   *If Active*: Update `session.last_audio_activity = Now`.
    *   *If Silent*: Check `(Now - session.last_audio_activity)`.
        *   If > `SILENCE_THRESHOLD` (10s), check Debounce (5s).
        *   **Log Violation**.

#### C. The Event Loop (Interrupts)
*   **Tab Switching**: The `document.visibilitychange` event fires immediately when the user switches tabs.
*   **Window Blur**: The `window.onblur` event fires when the user clicks a different application (e.g., NotePad, ChatGPT).
*   **Action**: These fire immediate POST requests to `/api/log_tab_switch/`.

### Phase 3: Graduation & Scoring
1.  **End Command**: User clicks "End Interview".
2.  **Aggregation**:
    *   Backend sets `is_active = False`.
    *   Backend retrieves **all** `ViolationLog` entries for this session.
    *   **Penalty Calculation**:
        *   Base Score: 100.
        *   Subtracts points based on `VIOLATION_PENALTIES` map.
        *   *Example*: 2 Tab Switches (-16) + 1 Face Missing (-5) = 100 - 21 = **79.0**.
3.  **Risk Assessment**:
    *   Score > 85: **Green**
    *   Score 50-85: **Yellow**
    *   Score < 50: **Red**
4.  **Result**: The final JSON response directs the frontend to show the score and redirect to the specific page.

## 💻 Technology Stack
*   **Language**: Python 3.x
*   **Web Framework**: Django
*   **Computer Vision**: OpenCV (`cv2`), NumPy
*   **Frontend**: JavaScript (ES6+), HTML5, CSS3
*   **Database**: SQLite (default setup)

## 🧠 Machine Learning & Algorithms

### 1. Face Detection (OpenCV)
The core visual monitoring uses **Haar Feature-based Cascade Classifiers**.
*   **Model**: `haarcascade_frontalface_default.xml`
*   **Process**:
    1.  Convert incoming Base64 image to NumPy array.
    2.  Convert BGR image to Grayscale (reduces computational load by 3x).
    3.  Apply `detectMultiScale`.
*   **Algorithmic Choices**:
    *   We use Haar Cascades over Deep Learning (like DNN or CNN) because it is **extremely lightweight**. It allows the server to handle multiple candidates 1FPS without needing a GPU.

### 2. Audio Processing (Web Audio API)
*   **FFT Analysis**: Uses `ScriptProcessorNode` to analyze raw PCM data.
*   **Smoothing**: A time constant of `0.8` is used to smooth out momentary spikes, preventing false positives from keyboard clicks.

## 📊 Scoring System

The candidate starts with **100 points**. Penalties are deducted as follows:

| Violation Type | Penalty |
| :--- | :--- |
| **Multiple Faces** | -10 points |
| **Tab Switch** | -8 points |
| **Face Missing** | -5 points |
| **Prolonged Silence** | -5 points |
| **Face Orientation** | -2 points |

**Risk Levels**:
*   🟢 **Green**: Score > 85 (Pass)
*   🟡 **Yellow**: Score > 50 (Review Needed)
*   🔴 **Red**: Score <= 50 (High Risk)
