from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from actions.browser_control import browser_control
from actions.code_helper import code_helper
from actions.computer_control import computer_control
from actions.computer_settings import computer_settings
from actions.desktop import desktop_control
from actions.dev_agent import dev_agent
from actions.file_controller import file_controller
from actions.file_processor import file_processor
from actions.flight_finder import flight_finder
from actions.game_updater import game_updater
from actions.open_app import open_app
from actions.proactive import ProactiveEngine
from actions.reminder import reminder
from actions.screen_processor import _capture_camera, _capture_screen
from actions.send_message import send_message
from actions.system_monitor import SystemMonitor, get_system_status
from actions.weather_report import weather_action
from actions.web_search import web_search as web_search_action
from actions.youtube_video import youtube_video
from core.installer import install_for_config
from core.llm_client import call_llm, check_model_available, ensure_ollama_running, warmup_model
from core.stt import WhisperSTT
from core.tts import create_tts_player
from memory.memory_manager import format_memory_for_prompt, load_memory, update_memory

CHANNELS = 1
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
VAD_THRESHOLD = 0.005
VAD_SILENCE_SEC = 0.8
MIN_UTTERANCE_SEC = 0.5


def _get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


BASE_DIR = _get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH = BASE_DIR / "core" / "prompt.txt"


def _load_runtime_config() -> dict:
    try:
        data = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    if data.get("gemini_api_key") and not data.get("llm_api_key"):
        data["llm_api_key"] = data["gemini_api_key"]

    data.setdefault("llm_provider", "openai")
    data.setdefault("llm_url", "https://api.xiaomimimo.com/v1")
    data.setdefault("llm_model", "mimo-v2.5")
    data.setdefault("llm_auth_mode", "bearer")
    data.setdefault("stt_engine", "whisper")
    data.setdefault("stt_model", "base")
    data.setdefault("stt_language", "en")
    data.setdefault("tts_engine", "edgetts")
    data.setdefault("tts_voice", "en-US-GuyNeural")
    data.setdefault("dashboard_enabled", False)
    data.setdefault("startup_briefing_enabled", False)
    return data


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks."
        )


def _clean_transcript(text: str) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split()).strip()


def _pick_input_device() -> int | None:
    """Return a usable input device index, preferring the OS default when valid."""
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None

    try:
        devices = sd.query_devices()
    except Exception:
        return None

    valid_inputs = [
        idx for idx, dev in enumerate(devices)
        if int(dev.get("max_input_channels", 0) or 0) > 0
    ]
    if not valid_inputs:
        return None
    if isinstance(default_in, int) and default_in in valid_inputs:
        return default_in
    return valid_inputs[0]


@dataclass
class FunctionCallStub:
    id: str
    name: str
    args: dict


class JarvisMimoLive:
    def __init__(self, ui, tool_declarations: list[dict]):
        self.ui = ui
        self._tool_declarations = tool_declarations
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dashboard = None
        self._config: dict = {}
        self._conversation: list[dict] = []
        self._turn_lock = asyncio.Lock()
        self._audio_input_queue: asyncio.Queue | None = None
        self._tts_player = None
        self._tts_queue: "queue.Queue[str | None]" = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()
        self._stt = None
        self._is_speaking = False
        self._speaking_lock = threading.Lock()
        self._phone_active = False
        self._pending_vision = None
        self._vision_cam_active = False
        self._vision_close_pending = False
        self._vision_last_time = 0.0
        self._vision_busy = False
        self._interrupted = False
        self._recording = False
        self._speech_chunks: list[np.ndarray] = []
        self._pre_roll: deque[np.ndarray] = deque(maxlen=8)
        self._last_voice_time = 0.0
        self._briefing_sent = False
        self._warmup_done = False
        self._sys_monitor = SystemMonitor()
        self._proactive = ProactiveEngine()
        self._last_user_speech = time.monotonic()
        self._manual_listen_armed = False
        self._wake_phrases = ("hey jarvis", "ok jarvis", "jarvis")
        self.ui.on_text_command = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        self.ui.on_voice_trigger = self.arm_voice_trigger
        self.ui.on_interrupt = self.interrupt

    def _load_components(self) -> None:
        self._config = _load_runtime_config()
        if self._config.get("llm_provider", "openai").lower() == "openai" and not self._config.get("llm_api_key"):
            raise RuntimeError("Missing llm_api_key in config/api_keys.json")
        self._stt = WhisperSTT(
            model_name=self._config.get("stt_model", "base"),
            language=self._config.get("stt_language"),
        )
        self._tts_player = create_tts_player(self._config)

    def _make_remote_key(self):
        if self._dashboard is None:
            self.ui.write_log(
                "SYS: Dashboard unavailable. "
                "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            )
            return None
        key = self._dashboard.new_key()
        url = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _tool_specs(self) -> list[dict]:
        return [{"type": "function", "function": spec} for spec in self._tool_declarations]

    def _build_system_instruction(self) -> str:
        memory = load_memory()
        mem_str = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()
        now = datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
        parts = [
            "[CURRENT DATE & TIME]",
            f"Right now it is: {now}",
            "Use this to calculate exact times for reminders.",
            "",
        ]
        if mem_str:
            parts.append(mem_str)
            parts.append("")
        parts.append(sys_prompt)
        return "\n".join(parts)

    def _build_messages(self, history: list[dict]) -> list[dict]:
        return [{"role": "system", "content": self._build_system_instruction()}, *history]

    def _trim_history(self, history: list[dict], limit: int = 24) -> list[dict]:
        return history[-limit:]

    def _normalise_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        normalised = []
        for idx, tool_call in enumerate(tool_calls or []):
            fn = tool_call.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            normalised.append({
                "id": tool_call.get("id") or f"tool_{int(time.time() * 1000)}_{idx}",
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
                "_args_dict": args,
            })
        return normalised

    def _build_vision_message(self, img_bytes: bytes, mime_type: str, question: str) -> dict:
        import base64

        b64 = base64.b64encode(img_bytes).decode("ascii")
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ],
        }

    def _on_text_command(self, text: str):
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(
            self._handle_user_text(text, log_user=True, store_in_history=True),
            self._loop,
        )

    def arm_voice_trigger(self) -> None:
        self._manual_listen_armed = True
        self.ui.write_log("SYS: Voice trigger armed. Speak your command.")

    def _extract_wake_command(self, text: str) -> tuple[bool, str]:
        lowered = " ".join(text.lower().strip().split())
        for phrase in self._wake_phrases:
            if lowered.startswith(phrase):
                remainder = text[len(text) - len(text.lstrip()):]
                clean = text.strip()
                cmd = clean[len(phrase):].lstrip(" ,.!?:;-")
                return True, cmd
        return False, text

    def _queue_tts(self, text: str) -> None:
        if text and self._tts_player:
            self._tts_queue.put(text)

    def _tts_worker(self) -> None:
        while True:
            text = self._tts_queue.get()
            if text is None:
                return
            if not self._tts_player:
                continue
            self._interrupted = False
            self._tts_player.speak(
                text,
                on_start=lambda: self.set_speaking(True),
                on_done=lambda: self.set_speaking(False),
            )

    async def _safe_dashboard_serve(self) -> None:
        if not self._dashboard:
            return
        try:
            await self._dashboard.serve()
        except BaseException as e:
            self.ui.write_log(f"SYS: Dashboard disabled - {str(e)[:120]}")
            self._dashboard = None

    async def _warmup_llm_background(self) -> None:
        if self._warmup_done:
            return
        try:
            self.ui.write_log("SYS: Warming up MiMo model in background...")
            await asyncio.to_thread(warmup_model, _load_system_prompt())
            self._warmup_done = True
            self.ui.write_log("SYS: MiMo model ready.")
        except Exception as e:
            self.ui.write_log(f"SYS: MiMo warmup skipped - {str(e)[:120]}")

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def interrupt(self) -> None:
        self._interrupted = True
        while True:
            try:
                self._tts_queue.get_nowait()
            except queue.Empty:
                break
        if self._tts_player:
            self._tts_player.stop()
        self._recording = False
        self._speech_chunks.clear()
        self._pre_roll.clear()
        self.set_speaking(False)
        self.ui.write_log("SYS: Interrupted - listening...")

    def speak(self, text: str):
        self._queue_tts(text)

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:160]
        self.ui.write_log(f"ERR: {tool_name} - {short}")
        self.speak(f"{tool_name} encountered an error. {short}")

    async def _emit_assistant_text(self, text: str) -> None:
        clean = _clean_transcript(text)
        if not clean:
            return
        self.ui.write_log(f"Jarvis: {clean}")
        if self._dashboard:
            await self._dashboard.broadcast({
                "type": "log",
                "speaker": "jarvis",
                "text": clean,
                "ts": datetime.now().isoformat(),
            })
        self._queue_tts(clean)

    async def _handle_user_text(
        self,
        text: str,
        *,
        log_user: bool,
        store_in_history: bool,
    ) -> None:
        clean = _clean_transcript(text)
        try:
            if not clean:
                return

            self._last_user_speech = time.monotonic()
            if log_user:
                self.ui.write_log(f"You: {clean}")
                if self._dashboard:
                    await self._dashboard.broadcast({
                        "type": "log",
                        "speaker": "user",
                        "text": clean,
                        "ts": datetime.now().isoformat(),
                    })

            await self._run_conversation_turn(
                {"role": "user", "content": clean},
                store_in_history=store_in_history,
            )
        except Exception as e:
            self.ui.write_log(f"ERR: Request failed - {str(e)[:160]}")
        finally:
            if not self.ui.muted and not self._is_speaking:
                self.ui.set_state("LISTENING")

    async def _run_conversation_turn(self, user_message: dict, *, store_in_history: bool) -> None:
        async with self._turn_lock:
            history = list(self._conversation)
            history.append(user_message)
            resolved = await self._resolve_conversation(history)
            if store_in_history:
                self._conversation = self._trim_history(resolved)

    async def _resolve_conversation(self, history: list[dict]) -> list[dict]:
        while True:
            self.ui.set_state("THINKING")
            response = await asyncio.to_thread(
                call_llm,
                self._build_messages(history),
                self._tool_specs(),
            )
            content = (response.get("content") or "").strip()
            tool_calls = self._normalise_tool_calls(response.get("tool_calls") or [])

            if content or tool_calls:
                assistant_msg: dict[str, object] = {"role": "assistant", "content": content}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": tc["type"],
                            "function": tc["function"],
                        }
                        for tc in tool_calls
                    ]
                history.append(assistant_msg)

            if content:
                await self._emit_assistant_text(content)

            if tool_calls:
                for tool_call in tool_calls:
                    name = tool_call["function"]["name"]
                    args = tool_call["_args_dict"]
                    result = await self._execute_tool(
                        FunctionCallStub(id=tool_call["id"], name=name, args=args)
                    )
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": json.dumps({"result": result}, ensure_ascii=False),
                    })
                continue

            if self._pending_vision:
                img_b, mime_t, question, angle = self._pending_vision
                self._pending_vision = None
                history.append(self._build_vision_message(img_b, mime_t, question))
                print(f"[Vision] Injecting {len(img_b):,} bytes (angle={angle}) into chat/completions")
                if self._vision_cam_active:
                    self._vision_cam_active = False
                    self._vision_close_pending = True
                else:
                    self._vision_busy = False
                continue

            if self._vision_close_pending:
                self._vision_close_pending = False
                self._vision_busy = False
                await asyncio.sleep(1.0)
                self.ui.stop_camera_stream()

            if not self.ui.muted and not self._is_speaking:
                self.ui.set_state("LISTENING")
            return history

    async def _execute_tool(self, fc: FunctionCallStub) -> str:
        name = fc.name
        args = dict(fc.args or {})
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key = args.get("key", "")
            value = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
            return "ok"

        loop = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "open_app":
                r = await loop.run_in_executor(None, lambda: open_app(parameters=args, response=None, player=self.ui))
                result = r or f"Opened {args.get('app_name')}."
            elif name == "weather_report":
                r = await loop.run_in_executor(None, lambda: weather_action(parameters=args, player=self.ui))
                result = r or "Weather delivered."
            elif name == "browser_control":
                r = await loop.run_in_executor(None, lambda: browser_control(parameters=args, player=self.ui))
                result = r or "Done."
            elif name == "file_controller":
                r = await loop.run_in_executor(None, lambda: file_controller(parameters=args, player=self.ui))
                result = r or "Done."
            elif name == "send_message":
                r = await loop.run_in_executor(None, lambda: send_message(parameters=args, response=None, player=self.ui, session_memory=None))
                result = r or f"Message sent to {args.get('receiver')}."
            elif name == "reminder":
                r = await loop.run_in_executor(None, lambda: reminder(parameters=args, response=None, player=self.ui))
                result = r or "Reminder set."
            elif name == "youtube_video":
                r = await loop.run_in_executor(None, lambda: youtube_video(parameters=args, response=None, player=self.ui))
                result = r or "Done."
            elif name == "screen_process":
                now = time.monotonic()
                cooldown = 4.0
                if self._vision_busy or (now - self._vision_last_time) < cooldown:
                    wait = max(0, cooldown - (now - self._vision_last_time))
                    result = f"Vision is still processing the previous request. Wait about {wait:.1f} seconds."
                else:
                    self._vision_busy = True
                    self._vision_last_time = now
                    angle = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    result = (
                        f"[VISION_ACTIVE] {stall.capitalize()} captured. "
                        "Acknowledge in one short sentence, then wait for the next message that contains the image."
                    )
            elif name == "close_camera":
                self.ui.stop_camera_stream()
                self._vision_busy = False
                result = "Camera closed."
            elif name == "computer_settings":
                r = await loop.run_in_executor(None, lambda: computer_settings(parameters=args, response=None, player=self.ui))
                result = r or "Done."
            elif name == "desktop_control":
                r = await loop.run_in_executor(None, lambda: desktop_control(parameters=args, player=self.ui))
                result = r or "Done."
            elif name == "code_helper":
                r = await loop.run_in_executor(None, lambda: code_helper(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."
            elif name == "dev_agent":
                r = await loop.run_in_executor(None, lambda: dev_agent(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."
            elif name == "web_search":
                r = await loop.run_in_executor(None, lambda: web_search_action(parameters=args, player=self.ui))
                result = r or "Done."
                mode = args.get("mode", "search")
                if r and not r.startswith("No results") and not r.startswith("Search failed"):
                    query = args.get("query") or ", ".join(args.get("items", []))
                    label = f"{mode.upper()} - {query[:38]}" if query else mode.upper()
                    self.ui.show_content(label, r)
            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = await loop.run_in_executor(None, lambda: file_processor(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."
            elif name == "computer_control":
                r = await loop.run_in_executor(None, lambda: computer_control(parameters=args, player=self.ui))
                result = r or "Done."
            elif name == "game_updater":
                r = await loop.run_in_executor(None, lambda: game_updater(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."
            elif name == "flight_finder":
                r = await loop.run_in_executor(None, lambda: flight_finder(parameters=args, player=self.ui))
                result = r or "Done."
            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)
            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                self.speak("Goodbye.")
                def _shutdown():
                    time.sleep(1)
                    import os
                    os._exit(0)
                threading.Thread(target=_shutdown, daemon=True).start()
                result = "Shutting down."
            else:
                result = f"Unknown tool: {name}"
        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted and not self._is_speaking:
            self.ui.set_state("LISTENING")
        return result

    async def _listen_audio(self):
        loop = asyncio.get_event_loop()
        input_device = _pick_input_device()

        if input_device is None:
            self.ui.write_log(
                "SYS: No microphone/input device detected by Windows. "
                "Voice input disabled; text commands still work."
            )
            while True:
                await asyncio.sleep(60)

        def _flush_utterance() -> None:
            if not self._recording or not self._speech_chunks or not self._audio_input_queue:
                self._recording = False
                self._speech_chunks.clear()
                return
            utterance = np.concatenate(self._speech_chunks)
            duration = len(utterance) / SEND_SAMPLE_RATE
            self._recording = False
            self._speech_chunks = []
            self._pre_roll.clear()
            if duration < MIN_UTTERANCE_SEC:
                return
            self.ui.write_log(f"SYS: Voice captured ({duration:.1f}s), sending to STT...")
            loop.call_soon_threadsafe(self._audio_input_queue.put_nowait, utterance.copy())

        def callback(indata, _frames, _time_info, status):
            if status:
                print(f"[Mic] {status}")

            with self._speaking_lock:
                jarvis_speaking = self._is_speaking

            chunk = np.copy(indata[:, 0]).astype(np.int16)
            level = float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2) + 1e-12))
            now = time.monotonic()

            if jarvis_speaking or self.ui.muted or self._phone_active:
                self._recording = False
                self._speech_chunks = []
                self._pre_roll.clear()
                return

            self._pre_roll.append(chunk)
            if level >= VAD_THRESHOLD:
                if not self._recording:
                    self._recording = True
                    self._speech_chunks = list(self._pre_roll)
                    self.ui.write_log("SYS: Voice detected.")
                else:
                    self._speech_chunks.append(chunk)
                self._last_voice_time = now
                return

            if self._recording:
                self._speech_chunks.append(chunk)
                if now - self._last_voice_time >= VAD_SILENCE_SEC:
                    _flush_utterance()

        try:
            with sd.InputStream(
                device=input_device,
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                self.ui.write_log(f"SYS: Microphone ready (device {input_device}).")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            self.ui.write_log(
                f"SYS: Microphone unavailable - {str(e)[:120]}. "
                "Voice input disabled; text commands still work."
            )
            while True:
                await asyncio.sleep(60)

    async def _transcribe_loop(self):
        while True:
            if not self._audio_input_queue:
                await asyncio.sleep(0.1)
                continue
            utterance = await self._audio_input_queue.get()
            if self._interrupted:
                self._interrupted = False
                continue
            self.ui.set_state("PROCESSING")
            try:
                audio = utterance.astype(np.float32) / 32768.0
                text = await asyncio.to_thread(self._stt.transcribe, audio)
                self.ui.write_log(f"SYS: STT result: {text[:80] or '[empty]'}")
                clean = _clean_transcript(text)
                if not clean:
                    if not self.ui.muted:
                        self.ui.set_state("LISTENING")
                    continue

                if self._manual_listen_armed:
                    self._manual_listen_armed = False
                    await self._handle_user_text(clean, log_user=True, store_in_history=True)
                    continue

                matched, command = self._extract_wake_command(clean)
                if not matched:
                    self.ui.write_log("SYS: Wake word not detected. Say 'hey jarvis' or use VOICE TRIGGER.")
                    if not self.ui.muted:
                        self.ui.set_state("LISTENING")
                    continue

                if not command:
                    self._manual_listen_armed = True
                    self.ui.write_log("SYS: Wake word detected. Listening for your command...")
                    if not self.ui.muted:
                        self.ui.set_state("LISTENING")
                    continue

                await self._handle_user_text(command, log_user=True, store_in_history=True)
            except Exception as e:
                self.ui.write_log(f"ERR: STT - {e}")
                if not self.ui.muted:
                    self.ui.set_state("LISTENING")

    async def _send_startup_briefing(self) -> None:
        await asyncio.sleep(0.3)
        try:
            memory = load_memory()
            identity = memory.get("identity", {})

            def _val(key: str) -> str:
                entry = identity.get(key, {})
                return (entry.get("value", "") if isinstance(entry, dict) else str(entry)).strip()

            lang = _val("language")
            name = _val("name")
            time_str = datetime.now().strftime("%H:%M")
            lang_clause = f" Respond in {lang}." if lang else ""
            name_clause = f" Address the user as {name}." if name else ""
            p1 = (
                f"Greet the user, mention it is {time_str}, and say you are fetching today's news headlines now. "
                f"One short sentence only. Do not call any tools.{lang_clause}{name_clause}"
            )
            await self._handle_user_text(p1, log_user=False, store_in_history=False)
        except Exception as e:
            self.ui.write_log(f"SYS: Startup briefing skipped - {str(e)[:140]}")
            return

        async def _guarded_news():
            try:
                await self._briefing_news_phase(lang)
            except Exception as e:
                self.ui.write_log(f"SYS: Briefing news phase failed: {e}")
        asyncio.create_task(_guarded_news())

    async def _briefing_news_phase(self, lang: str) -> None:
        await asyncio.sleep(1.5)
        lang_str = f" Respond in {lang}." if lang else ""
        p2 = (
            "[BRIEFING] Call web_search with mode='news' and query='top world news today' "
            "to find actual recent news articles with real event headlines. "
            "After the search, say one specific news event from the results in one sentence, "
            f"then say the full list is displayed on screen.{lang_str}"
        )
        await self._handle_user_text(p2, log_user=False, store_in_history=False)

    async def _run_system_monitor(self) -> None:
        while True:
            await asyncio.sleep(10)
            alert = await asyncio.to_thread(self._sys_monitor.check)
            if alert:
                try:
                    await self._handle_user_text(alert, log_user=False, store_in_history=False)
                except Exception as e:
                    print(f"[Monitor] Could not send alert: {e}")

    async def _run_proactive_mode(self) -> None:
        while True:
            await asyncio.sleep(60)
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue
            if not self._proactive.should_trigger(self._last_user_speech):
                continue
            self._proactive.mark_triggered()
            try:
                memory = await asyncio.to_thread(load_memory)
                prompt = self._proactive.build_prompt(memory)
                await self._handle_user_text(prompt, log_user=False, store_in_history=False)
                self.ui.write_log("SYS: Proactive check-in.")
            except Exception as e:
                print(f"[Proactive] {e}")

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    async def _process_dashboard_commands(self) -> None:
        while True:
            if not self._dashboard:
                await asyncio.sleep(0.5)
                continue
            try:
                text = await asyncio.wait_for(self._dashboard._command_queue.get(), timeout=0.5)
                if not text:
                    continue
                await self._handle_user_text(text, log_user=False, store_in_history=True)
                self.ui.write_log(f"[Web]: {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    async def run(self):
        self._loop = asyncio.get_event_loop()

        while True:
            try:
                self.ui.set_state("THINKING")
                self.ui.write_log("SYS: Loading configuration...")
                self._config = _load_runtime_config()
                if self._config.get("dashboard_enabled"):
                    try:
                        from dashboard.server import DashboardServer

                        self._dashboard = DashboardServer()
                        self._dashboard.set_connect_callback(self._on_phone_connected)
                        asyncio.create_task(self._safe_dashboard_serve())
                        asyncio.create_task(self._process_dashboard_commands())
                    except Exception as e:
                        self.ui.write_log(f"SYS: Dashboard disabled - {str(e)[:120]}")
                        self._dashboard = None
                self.ui.write_log("SYS: Checking dependencies...")
                await asyncio.to_thread(install_for_config, self._config, self.ui.write_log)
                self.ui.write_log("SYS: Loading speech engines...")
                await asyncio.to_thread(self._load_components)

                if self._config.get("llm_provider", "openai").lower() == "ollama":
                    self.ui.write_log("SYS: Starting Ollama...")
                    await asyncio.to_thread(ensure_ollama_running)
                    await asyncio.to_thread(check_model_available, self.ui.write_log)

                self._audio_input_queue = asyncio.Queue()
                self._pending_vision = None
                self._vision_cam_active = False
                self._vision_close_pending = False
                self._vision_busy = False
                self._vision_last_time = 0.0
                self._interrupted = False

                self.ui.set_state("LISTENING")
                self.ui.write_log("SYS: JARVIS online.")
                if self._dashboard:
                    await self._dashboard.broadcast({"type": "status", "state": "active"})
                asyncio.create_task(self._warmup_llm_background())

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._transcribe_loop())
                    tg.create_task(self._run_system_monitor())
                    tg.create_task(self._run_proactive_mode())
                    if self._config.get("startup_briefing_enabled") and not self._briefing_sent:
                        self._briefing_sent = True
                        tg.create_task(self._send_startup_briefing())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                err_str = str(e)
                print(f"[JARVIS] Error ({type(e).__name__}): {e}")
                traceback.print_exc()

                if any(token in err_str for token in ("401", "403", "Unauthorized", "invalid_api_key")):
                    self.ui.write_log("ERR: API key rejected - please re-enter your MiMo/OpenAI key and base URL.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    continue

                self.ui.write_log(f"ERR: {err_str[:180]}")
            finally:
                self.set_speaking(False)
                self.ui.set_state("SLEEPING")
                if self._dashboard:
                    await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            await asyncio.sleep(3)
