# Arad

An AI-powered desktop assistant for voice interaction, system control, screen analysis, browser automation, file handling, and remote mobile access.

**Author:** Ali Mohammadabadi

---

## Overview

`Arad` is a Python-based desktop assistant designed to combine natural interaction with practical desktop automation. It can listen, speak, inspect the screen or camera, search the web, manage files, control apps, and expose a remote dashboard for mobile control.

The project uses `PyQt6` for the desktop interface and is organized into modular components for speech, text generation, browser control, system utilities, memory, and dashboard services.

---

## Features

### Core Assistant

- Voice interaction with `Whisper STT` and `EdgeTTS`
- Support for `OpenAI-compatible` LLM providers through configuration
- Persistent memory for user context and long-term preferences
- Fast interrupt handling with `ESC` or the `INTERRUPT` button
- Session-based runtime flow for listening, thinking, and responding

### System and Desktop Control

- Launch desktop applications and tools
- Control volume, brightness, Wi-Fi, restart, and shutdown actions
- Trigger shortcuts, type text, and control mouse and keyboard input
- Manage windows, tabs, scrolling, and general desktop actions
- Monitor CPU, RAM, and system health with background alerts

### Vision and Media

- Capture and analyze the current screen
- Capture webcam input and respond based on visual context
- Built-in cooldown protection for repeated vision requests
- YouTube search, playback, and summary support

### Web and Automation

- Web search modes: `search`, `news`, `research`, `price`, and `compare`
- Startup briefing and live news lookup
- Browser automation for opening pages, clicking, typing, scrolling, and screenshots
- Weather lookup, reminders, flight search, and messaging actions

### Files and Developer Tools

- Read, write, create, delete, move, copy, and inspect files
- Process uploaded or local files
- Summarize documents and answer questions about their contents
- Built-in coding helper for writing, editing, running, and explaining code
- `dev_agent` support for larger development workflows

### Remote Dashboard

- `FastAPI`-based web dashboard
- Mobile pairing with QR code
- Send commands from phone to desktop
- Stream phone microphone audio into the assistant
- Upload and download files through the dashboard
- Reconnect known devices with persistent session handling

---

## What's New in This Build

- Lower-latency speech interruption
- Improved screen and camera response flow
- Better startup briefing behavior
- More structured web and news search handling
- Cleaner session state isolation across reconnects
- Stronger support for mobile dashboard control
- Proactive behavior during long user silence

---

## Project Structure

```text
Mark-XLVIII-main/
|-- main.py
|-- ui.py
|-- setup.py
|-- requirements.txt
|-- readme.md
|-- actions/
|   |-- browser_control.py
|   |-- code_helper.py
|   |-- computer_control.py
|   |-- computer_settings.py
|   |-- desktop.py
|   |-- dev_agent.py
|   |-- file_controller.py
|   |-- file_processor.py
|   |-- flight_finder.py
|   |-- game_updater.py
|   |-- open_app.py
|   |-- proactive.py
|   |-- reminder.py
|   |-- screen_processor.py
|   |-- send_message.py
|   |-- system_monitor.py
|   |-- weather_report.py
|   |-- web_search.py
|   `-- youtube_video.py
|-- core/
|   |-- installer.py
|   |-- llm_client.py
|   |-- mimo_runtime.py
|   |-- prompt.txt
|   |-- stt.py
|   `-- tts.py
|-- dashboard/
|   |-- server.py
|   `-- static/
|-- memory/
|   |-- config_manager.py
|   |-- long_term.json
|   `-- memory_manager.py
`-- config/
    |-- api_keys.json
    `-- certs/
```

---

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run setup

```bash
python setup.py
```

### 3. Configure the assistant

Edit `config/api_keys.json` with your preferred API key, model, speech settings, and runtime options.

### 4. Start the app

```bash
python main.py
```

---

## Requirements

- Python `3.11+`
- A microphone for voice interaction
- Windows, Linux, or macOS
- An installed browser for automation features
- An `OpenAI-compatible` API endpoint or other configured LLM backend

---

## Key Dependencies

- `PyQt6` for the desktop UI
- `sounddevice` and `numpy` for audio handling
- `playwright` for browser automation
- `fastapi` and `uvicorn` for the remote dashboard
- `opencv-python` and `mss` for screen and camera features
- `psutil` for system monitoring

---

## Notes

- Some system-control features are more complete on Windows
- Browser automation may require Playwright browser installation
- Remote dashboard features depend on `fastapi`, `uvicorn`, `cryptography`, and `python-multipart`
- Sensitive keys inside `config/api_keys.json` should not be committed publicly

---

## Status

This README has been refreshed to match the current project structure and feature set while renaming the assistant from `MARK XLVIII` to `Arad`.
