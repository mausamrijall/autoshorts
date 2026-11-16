#!/usr/bin/env python3
"""
AutoShorts pipeline — Python port of src-tauri/src/{media,transcription,llm}.rs
=====================================================================================================================================
This module lets the core AutoShorts pipeline run outside the Tauri desktop
app — most notably inside Google Colab where a free NVIDIA GPU can be used
for local Whisper transcription.

The logic mirrors the Rust backend faithfully so results match the desktop app:

    1. probe_media()          -> ffprobe metadata (duration, video/audio, size)
    2. extract_audio()        -> ffmpeg 16 kHz mono WAV for transcription
    3. transcribe_local()     -> openai-whisper on CUDA (falls back to CPU)
       transcribe_deepgram()  -> Deepgram cloud transcription (optional)
    4. detect_candidates_*()  -> LLM viral-moment detection (DeepSeek, Claude,
                                  Gemini, OpenAI, Groq, OpenRouter, Ollama)
    5. generate_srt() / build_drawtext_filters() -> caption helpers
    6. render_flat_clip()     -> ffmpeg 9:16 vertical crop render

Network calls import `requests` lazily; Whisper imports `whisper`/`torch`
lazily — so the module can be imported (and its pure functions unit-tested)
even where those dependencies are absent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True)


def require_ffmpeg() -> None:
    if not command_exists("ffmpeg") or not command_exists("ffprobe"):
        raise RuntimeError(
            "ffmpeg/ffprobe are not installed or not on PATH. "
            "In Colab: !apt-get install -y ffmpeg"
        )


# ---------------------------------------------------------------------------
# Media probing & audio extraction (port of media.rs)
# ---------------------------------------------------------------------------

def probe_media(path: str) -> dict[str, Any]:
    """Return the same MediaProbe shape as src-tauri/src/media.rs."""
    require_ffmpeg()
    result = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ])
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed: " + result.stderr.decode(errors="replace").strip())

    data = json.loads(result.stdout.decode())
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration_sec = None
    if data.get("format", {}).get("duration"):
        try:
            duration_sec = float(data["format"]["duration"])
        except (TypeError, ValueError):
            duration_sec = None

    return {
        "duration_sec": duration_sec,
        "has_video": video is not None,
        "width": video.get("width") if video else None,
        "height": video.get("height") if video else None,
        "video_codec": video.get("codec_name") if video else None,
        "audio_codec": audio.get("codec_name") if audio else None,
    }


def extract_audio(source_path: str, project_dir: str | Path) -> str:
    """Extract mono 16 kHz WAV, returns path to the WAV file (media.rs)."""
    require_ffmpeg()
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    output_path = project_dir / "transcription_audio.wav"

    result = _run([
        "ffmpeg", "-y", "-i", source_path, "-vn", "-ac", "1", "-ar", "16000",
        str(output_path),
    ])
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio extraction failed: "
            + result.stderr.decode(errors="replace").strip()
        )
    return str(output_path)


# ---------------------------------------------------------------------------
# Transcription (port of transcription.rs)
# ---------------------------------------------------------------------------

def _normalize_deepgram(value: dict[str, Any]) -> dict[str, Any]:
    channels = value.get("results", {}).get("channels", [])
    if not channels:
        raise ValueError("Deepgram response did not include a channel")
    alternative = channels[0].get("alternatives", [{}])[0]
    metadata = value.get("metadata", {})

    language = metadata.get("language", "en")
    try:
        duration = float(metadata.get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    speakers: set[str] = set()
    words: list[dict[str, Any]] = []
    for w in alternative.get("words", []):
        text = (w.get("punctuated_word") or w.get("word") or "").strip()
        if not text:
            continue
        speaker = w.get("speaker")
        speaker_tag = f"S{speaker + 1}" if isinstance(speaker, int) else None
        if speaker_tag:
            speakers.add(speaker_tag)
        words.append({
            "text": text,
            "start": float(w.get("start", 0.0)),
            "end": float(w.get("end", 0.0)),
            "speaker": speaker_tag,
        })

    segments = build_segments(words)
    return {
        "language": language,
        "duration": duration,
        "speakers": sorted(speakers),
        "words": words,
        "segments": segments,
    }


def transcribe_deepgram(audio_path: str, api_key: str) -> dict[str, Any]:
    """Cloud transcription via Deepgram (deepgram nova-2, smart format)."""
    import requests  # lazy

    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()

    resp = requests.post(
        "https://api.deepgram.com/v1/listen?"
        "model=nova-2&smart_format=true&diarize=true&punctuate=true&filler_words=true",
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
        data=audio_bytes,
        timeout=600,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Deepgram request failed ({resp.status_code}): {resp.text}")
    return _normalize_deepgram(resp.json())


def build_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Port of build_segments in transcription.rs."""
    segments: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None

    for word in words:
        should_break = False
        if current is not None:
            pause = word["start"] - current["end"]
            speaker_changed = current.get("speaker") != word.get("speaker")
            sentence_end = current["text"].rstrip().endswith((".", "!", "?"))
            should_break = pause > 0.9 or speaker_changed or sentence_end

        if should_break:
            segments.append(current)
            current = None

        if current is None:
            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker": word.get("speaker"),
                "text": word["text"],
            }
        else:
            current["end"] = word["end"]
            current["text"] += " " + word["text"]

    if current is not None:
        segments.append(current)
    return segments


def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def transcribe_local(
    audio_path: str,
    model_size: str = "base",
    device: Optional[str] = None,
) -> dict[str, Any]:
    """Local Whisper transcription using openai-whisper.

    The model runs on CUDA when a GPU is available (Colab/T4 etc) and falls
    back to CPU otherwise. Word-level timestamps are enabled so segments and
    caption rendering work exactly like the desktop app.
    """
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        ) from exc

    if device is None:
        device = _pick_device()

    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path, word_timestamps=True)

    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for seg in result.get("segments", []):
        start = float(seg["start"])
        end = float(seg["end"])
        text = seg.get("text", "").strip()
        segments.append({
            "start": start,
            "end": end,
            "speaker": "S1",
            "text": text,
        })
        for w in seg.get("words", []):
            words.append({
                "text": w.get("word", "").strip(),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "speaker": "S1",
            })

    duration = segments[-1]["end"] if segments else 0.0
    return {
        "language": result.get("language", "en"),
        "duration": duration,
        "speakers": ["S1"],
        "words": words,
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# LLM moment detection (port of llm.rs)
# ---------------------------------------------------------------------------

# Shared prompt for cloud providers — verbatim from src-tauri/src/llm.rs.
CLOUD_PROMPT = (
    "You are an elite, world-class social media strategist with a track record of "
    "generating viral multi-million-view Shorts, TikToks, and Reels. "
    "Your sole objective is to identify the ABSOLUTE BEST, most highly-engaging, "
    "and trend-setting short-form clip candidates from the provided transcript. "
    "Do NOT pick random or mediocre segments. Be ruthless in your selection, but "
    "extract AS MANY highly viral moments as possible. "
    "Every candidate must have an insanely strong, curiosity-inducing hook in the "
    "first 3 seconds to stop the scroll. "
    "Clips should be 30-90 seconds long, completely self-contained, cut at clean "
    "boundaries, and deliver a massive payoff (a mind-blowing fact, hilarious joke, "
    "highly controversial opinion, or deep emotional insight). "
    'Return up to 25 candidates as JSON matching exactly this schema: '
    '{"candidates":[{"start":0.0,"end":0.0,"score":0.0,"hook":"...","rationale":"..."}]}'
)

SYSTEM_OLLAMA_PROMPT = (
    "You are an elite, world-class social media strategist with a track record of "
    "generating viral multi-million-view Shorts, TikToks, and Reels. "
    "Your sole objective is to identify the ABSOLUTE BEST, most highly-engaging, "
    "and trend-setting short-form clip candidates from the provided transcript. "
    "Do NOT pick random or mediocre segments. Be ruthless in your selection. "
    "Every candidate must have an insanely strong, curiosity-inducing hook in the "
    "first 3 seconds to stop the scroll. "
    "CRITICAL: Each clip candidate MUST have a duration between 30 and 90 seconds "
    "(i.e. 'end' minus 'start' must be between 30.0 and 90.0). "
    "Do NOT return short clips of less than 30 seconds. Combine multiple adjacent "
    "sentences to build a meaningful segment of 30-90 seconds. "
    "Favor highly shareable content: concrete stories, strong opinions, emotional "
    "turns, surprising or counter-intuitive claims, clear payoffs, and high-energy/"
    "dramatic peaks. "
    "You MUST identify and return at least 3-10 candidates. Do not return an empty "
    "candidates list. "
    "Ensure the 'start' and 'end' values correspond to actual timestamps in the "
    "transcript. Do not output 0.0 for start and end times."
)

OLLAMA_FORMAT = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "score": {"type": "number"},
                    "hook": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["start", "end", "score", "hook", "rationale"],
            },
        }
    },
    "required": ["candidates"],
}


def compact_segments(segments: list[dict[str, Any]]) -> str:
    """Port of compact_segments in llm.rs."""
    lines = []
    for seg in segments:
        speaker = seg.get("speaker") or "Speaker"
        lines.append(f"[{seg['start']:.2f}-{seg['end']:.2f}] {speaker}: {seg['text']}")
    return "\n".join(lines)


def parse_candidate_json(text: str, min_duration: float) -> list[dict[str, Any]]:
    """Port of parse_candidate_json in llm.rs (same normalization/filter/sort)."""
    trimmed = text.strip()
    if trimmed.startswith("```json"):
        trimmed = trimmed[len("```json"):].strip()
    elif trimmed.startswith("```"):
        trimmed = trimmed[len("```"):].strip()
    if trimmed.endswith("```"):
        trimmed = trimmed[:-3].strip()

    val = json.loads(trimmed)

    candidates_arr = None
    if isinstance(val, list):
        candidates_arr = val
    elif isinstance(val, dict):
        found = None
        for key in ("candidates", "Candidates", "moments", "clips", "segments", "results"):
            if isinstance(val.get(key), list):
                found = val[key]
                break
        if found is None:
            for value in val.values():
                if isinstance(value, list):
                    found = value
                    break
        if found is not None:
            candidates_arr = found
        elif "start" in val and "end" in val:
            candidates_arr = [val]

    if candidates_arr is None:
        raise ValueError(f"LLM output does not contain a candidates array: {trimmed[:300]}")

    drafts: list[dict[str, Any]] = []
    for item in candidates_arr:
        start = _to_f64(item.get("start"), 0.0)
        end = _to_f64(item.get("end"), 0.0)
        score = _to_f64(item.get("score"), 0.8)

        if 10.0 < score <= 100.0:
            score /= 100.0
        elif 1.0 < score <= 10.0:
            score /= 10.0
        elif score > 100.0:
            score = 1.0
        elif score < 0.0:
            score = 0.0

        hook = item.get("hook") or ""
        rationale = item.get("rationale") or ""
        drafts.append({
            "start": start,
            "end": end,
            "score": score,
            "hook": hook,
            "rationale": rationale,
        })

    candidates = [
        c for c in drafts
        if (c["end"] - c["start"]) >= min_duration and c["hook"].strip()
    ]
    if not candidates:
        candidates = [
            c for c in drafts
            if (c["end"] - c["start"]) >= 5.0 and c["hook"].strip()
        ]

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:10]


def _to_f64(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _min_duration(transcript: dict[str, Any]) -> float:
    duration = float(transcript.get("duration", 0.0))
    if duration < 60.0:
        return max(5.0, duration * 0.5)
    return 30.0


def _user_content(transcript: dict[str, Any]) -> str:
    return "Transcript:\n" + compact_segments(transcript.get("segments", []))


# --- Cloud providers --------------------------------------------------------

def detect_candidates_with_deepseek(
    transcript: dict[str, Any],
    api_key: str,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    import requests  # lazy
    model = model_name or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": _user_content(transcript)}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek request failed ({resp.status_code}): {resp.text}")
    text = resp.json()["choices"][0]["message"]["content"]
    return parse_candidate_json(text, _min_duration(transcript))


def detect_candidates_with_claude(
    transcript: dict[str, Any],
    api_key: str,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    import requests  # lazy
    model = model_name or os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest"
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1800,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": _user_content(transcript)}],
        },
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Claude request failed ({resp.status_code}): {resp.text}")
    text = next(
        (b.get("text") for b in resp.json().get("content", []) if b.get("text")),
        "",
    )
    return parse_candidate_json(text, _min_duration(transcript))


def detect_candidates_with_gemini(
    transcript: dict[str, Any],
    api_key: str,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    import requests  # lazy
    model = model_name or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    resp = requests.post(
        url,
        json={
            "contents": [
                {"parts": [{"text": _user_content(transcript)}]}
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
            },
        },
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini request failed ({resp.status_code}): {resp.text}")
    candidates = resp.json().get("candidates", [])
    text = candidates[0]["content"]["parts"][0].get("text", "")
    return parse_candidate_json(text, _min_duration(transcript))


def detect_candidates_with_openai(
    transcript: dict[str, Any],
    api_key: str,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    import requests  # lazy
    model = model_name or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": _user_content(transcript)}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI request failed ({resp.status_code}): {resp.text}")
    text = resp.json()["choices"][0]["message"]["content"]
    return parse_candidate_json(text, _min_duration(transcript))


def detect_candidates_with_groq(
    transcript: dict[str, Any],
    api_key: str,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    import requests  # lazy
    model = model_name or os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile"
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": _user_content(transcript)}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Groq request failed ({resp.status_code}): {resp.text}")
    text = resp.json()["choices"][0]["message"]["content"]
    return parse_candidate_json(text, _min_duration(transcript))


def detect_candidates_with_openrouter(
    transcript: dict[str, Any],
    api_key: str,
    model_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    import requests  # lazy
    model = (
        model_name
        or os.environ.get("OPENROUTER_MODEL")
        or "google/gemini-2.5-flash"
    )
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": _user_content(transcript)}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter request failed ({resp.status_code}): {resp.text}")
    text = resp.json()["choices"][0]["message"]["content"]
    return parse_candidate_json(text, _min_duration(transcript))


def detect_candidates_with_local_llm(
    transcript: dict[str, Any],
    model_name: str,
    ollama_host: str = "http://localhost:11434",
) -> list[dict[str, Any]]:
    """Local Ollama model (GPU-accelerated when Ollama uses CUDA)."""
    import requests  # lazy
    resp = requests.post(
        f"{ollama_host}/api/chat",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_OLLAMA_PROMPT},
                {"role": "user", "content": _user_content(transcript)},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
            "format": OLLAMA_FORMAT,
        },
        timeout=600,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Local Ollama request failed ({resp.status_code}): {resp.text}")
    text = resp.json()["message"]["content"]
    return parse_candidate_json(text, _min_duration(transcript))


# ---------------------------------------------------------------------------
# Caption helpers (port of lib.rs generate_srt / build_drawtext_filters)
# ---------------------------------------------------------------------------

def format_srt_time(start: float, end: float) -> str:
    def fmt(secs: float) -> str:
        hours = int(secs // 3600)
        mins = int((secs % 3600) // 60)
        secs_only = int(secs % 60)
        ms = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{mins:02d}:{secs_only:02d},{ms:03d}"
    return f"{fmt(start)} --> {fmt(end)}"


def generate_srt(words: list[dict[str, Any]], start_sec: float, end_sec: float) -> str:
    """Port of generate_srt in lib.rs (chunks of 3 words)."""
    candidate_words = [w for w in words if w["end"] > start_sec and w["start"] < end_sec]

    out: list[str] = []
    index = 1
    for i in range(0, len(candidate_words), 3):
        chunk = candidate_words[i:i + 3]
        if not chunk:
            continue
        first, last = chunk[0], chunk[-1]
        start_rel = max(first["start"] - start_sec, 0.0)
        end_rel = min(last["end"] - start_sec, end_sec - start_sec)
        end_rel = max(end_rel, 0.0)
        text = " ".join(w["text"] for w in chunk)
        out.append(f"{index}\n{format_srt_time(start_rel, end_rel)}\n{text}\n")
        index += 1
    return "\n".join(out)


_FONT_PATHS = [
    # macOS
    "/System/Library/Fonts/Supplemental/Futura.ttc",
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    # Windows
    "C:/Windows/Fonts/SegoeUIb.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/SegoeUI.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    # Linux / Colab
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def _pick_font_option() -> str:
    for path in _FONT_PATHS:
        if os.path.exists(path):
            escaped = path.replace("\\", "/").replace("'", "'\\''")
            return f"fontfile='{escaped}':"
    return ""


def build_drawtext_filters(
    words: list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
    cropped_width: int,
    caption_style: str = "modern-box",
) -> str:
    """Port of build_drawtext_filters in lib.rs (chunks of 2 words)."""
    candidate_words = [w for w in words if w["end"] > start_sec and w["start"] < end_sec]
    if not candidate_words:
        return ""

    font_option = _pick_font_option()
    filters: list[str] = []

    for i in range(0, len(candidate_words), 2):
        chunk = candidate_words[i:i + 2]
        if not chunk:
            continue
        first, last = chunk[0], chunk[-1]
        start_rel = max(first["start"] - start_sec, 0.0)
        end_rel = min(last["end"] - start_sec, end_sec - start_sec)
        end_rel = max(end_rel, 0.0)
        if end_rel <= start_rel:
            continue

        text = " ".join(w["text"].upper() for w in chunk)
        clean_text = "".join(
            ch for ch in text if ch.isalnum() or ch in " !?"
        )

        fontsize = int(min(max(cropped_width * 0.075, 16.0), 80.0))
        padding = int(min(max(fontsize * 0.3, 4.0), 24.0))

        if caption_style == "classic-outline":
            borderw = int(min(max(fontsize * 0.1, 2.0), 8.0))
            filters.append(
                f"drawtext={font_option}text='{clean_text}':x=(w-text_w)/2:y=h*0.65:"
                f"fontsize={fontsize}:fontcolor=yellow:borderw={borderw}:bordercolor=black:"
                f"enable='between(t,{start_rel:.3f},{end_rel:.3f})'"
            )
        elif caption_style == "minimal-shadow":
            filters.append(
                f"drawtext={font_option}text='{clean_text}':x=(w-text_w)/2:y=h*0.7:"
                f"fontsize={fontsize}:fontcolor=white:shadowcolor=black@0.5:shadowx=2:shadowy=2:"
                f"enable='between(t,{start_rel:.3f},{end_rel:.3f})'"
            )
        elif caption_style == "vibrant-cyan":
            filters.append(
                f"drawtext={font_option}text='{clean_text}':x=(w-text_w)/2:y=h*0.7:"
                f"fontsize={fontsize}:fontcolor=0x00FFFF:shadowcolor=black@0.6:shadowx=2:shadowy=2:"
                f"enable='between(t,{start_rel:.3f},{end_rel:.3f})'"
            )
        elif caption_style == "vibrant-yellow-box":
            filters.append(
                f"drawtext={font_option}text='{clean_text}':x=(w-text_w)/2:y=h*0.72:"
                f"fontsize={fontsize}:fontcolor=black:box=1:boxcolor=0xffff00e0:boxborderw={padding}:"
                f"enable='between(t,{start_rel:.3f},{end_rel:.3f})'"
            )
        elif caption_style == "vibrant-green":
            borderw = int(min(max(fontsize * 0.08, 1.5), 6.0))
            filters.append(
                f"drawtext={font_option}text='{clean_text}':x=(w-text_w)/2:y=h*0.7:"
                f"fontsize={fontsize}:fontcolor=0x39FF14:borderw={borderw}:bordercolor=black:"
                f"shadowcolor=black@0.6:shadowx=2:shadowy=2:"
                f"enable='between(t,{start_rel:.3f},{end_rel:.3f})'"
            )
        else:  # modern-box (default)
            filters.append(
                f"drawtext={font_option}text='{clean_text}':x=(w-text_w)/2:y=h*0.72:"
                f"fontsize={fontsize}:fontcolor=white:box=1:boxcolor=0x000000b0:boxborderw={padding}:"
                f"enable='between(t,{start_rel:.3f},{end_rel:.3f})'"
            )

    return ",".join(filters)


# ---------------------------------------------------------------------------
# Clip rendering (port of media.rs render_flat_clip)
# ---------------------------------------------------------------------------

def render_flat_clip(
    source_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str | Path,
    drawtext_filters: Optional[str] = None,
) -> str:
    """Render a 9:16 vertical clip with optional burned-in captions."""
    require_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = f"{start_sec:.3f}"
    duration = f"{max(end_sec - start_sec, 0.1):.3f}"

    probe = probe_media(source_path)
    has_video = bool(probe.get("has_video"))

    cmd = ["ffmpeg", "-y", "-ss", start, "-i", source_path, "-t", duration]

    if has_video:
        filter_expr = (
            "crop=w='2*trunc(min(iw,ih*9/16)/2)':"
            "h='2*trunc(min(ih,iw*16/9)/2)'"
        )
        if drawtext_filters:
            filter_expr = f"{filter_expr},{drawtext_filters}"
        cmd += ["-vf", filter_expr]
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-vn"]

    cmd += ["-c:a", "aac", "-b:a", "192k", str(output_path)]

    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg clip render failed: "
            + result.stderr.decode(errors="replace").strip()
        )
    return str(output_path)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source_path: str,
    *,
    transcription: str = "local",   # "local" (Whisper GPU) or "deepgram"
    transcription_api_key: Optional[str] = None,
    whisper_model: str = "base",
    provider: str = "deepseek",      # deepseek|claude|gemini|openai|groq|openrouter|ollama
    provider_api_key: Optional[str] = None,
    provider_model: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    workdir: str | Path = ".",
    caption_style: str = "modern-box",
    num_clips: int = 5,
    render_clips: bool = True,
) -> dict[str, Any]:
    """Run ingest -> transcribe -> detect -> render. Returns a result dict."""
    workdir = Path(workdir)
    project_dir = workdir / "project"
    clips_dir = workdir / "clips"
    project_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    probe = probe_media(source_path)
    print(f"[1/4] Probed   {Path(source_path).name}: "
          f"duration={probe['duration_sec']}s video={probe['has_video']} "
          f"{probe['width']}x{probe['height']}")

    audio_path = extract_audio(source_path, project_dir)
    print(f"[2/4] Extracted audio -> {audio_path}")

    if transcription == "deepgram":
        if not transcription_api_key:
            raise ValueError("transcription='deepgram' requires transcription_api_key")
        transcript = transcribe_deepgram(audio_path, transcription_api_key)
        engine = "deepgram"
    else:
        transcript = transcribe_local(audio_path, model_size=whisper_model)
        engine = f"whisper/{whisper_model}({_pick_device()})"
    print(f"[3/4] Transcribed via {engine}: "
          f"{len(transcript['words'])} words, {len(transcript['segments'])} segments")

    if provider == "ollama":
        candidates = detect_candidates_with_local_llm(
            transcript, provider_model or "qwen2.5:7b", ollama_host
        )
    else:
        if not provider_api_key:
            raise ValueError(f"provider={provider!r} requires provider_api_key")
        fn = {
            "deepseek": detect_candidates_with_deepseek,
            "claude": detect_candidates_with_claude,
            "gemini": detect_candidates_with_gemini,
            "openai": detect_candidates_with_openai,
            "groq": detect_candidates_with_groq,
            "openrouter": detect_candidates_with_openrouter,
        }[provider]
        candidates = fn(transcript, provider_api_key, provider_model)
    print(f"[4/4] Detected {len(candidates)} viral clip candidates via {provider}")

    rendered: list[dict[str, Any]] = []
    for rank, cand in enumerate(candidates[:num_clips], start=1):
        out_path = clips_dir / f"clip-{rank:02d}_flat.mp4"
        try:
            drawtext = build_drawtext_filters(
                transcript["words"],
                cand["start"],
                cand["end"],
                _cropped_width(probe),
                caption_style,
            ) if render_clips else None
            render_flat_clip(
                source_path, cand["start"], cand["end"], out_path, drawtext
            )
            rendered.append({"rank": rank, **cand, "path": str(out_path)})
            print(f"    rendered clip-{rank:02d}_flat.mp4 ({cand['start']:.1f}-"
                  f"{cand['end']:.1f}s, score={cand['score']:.2f})")
        except Exception as exc:  # fall back to clean render, like the desktop app
            print(f"    caption render failed for rank {rank}: {exc}; retrying clean")
            render_flat_clip(source_path, cand["start"], cand["end"], out_path)
            rendered.append({"rank": rank, **cand, "path": str(out_path)})

    return {
        "probe": probe,
        "transcript": transcript,
        "engine": engine,
        "candidates": candidates,
        "clips": rendered,
        "project_dir": str(project_dir),
        "clips_dir": str(clips_dir),
    }


def _cropped_width(probe: dict[str, Any]) -> int:
    iw = float(probe.get("width") or 1920)
    ih = float(probe.get("height") or 1080)
    return int((min(iw, ih * 9.0 / 16.0) / 2.0) * 2.0)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="autoshorts_pipeline",
        description="AutoShorts pipeline (Colab/GPU friendly).",
    )
    parser.add_argument("source", help="Path to the media file (video or audio)")
    parser.add_argument("--workdir", default=".", help="Working/output directory")
    parser.add_argument("--transcription", choices=["local", "deepgram"], default="local")
    parser.add_argument("--deepgram-key", default=os.environ.get("DEEPGRAM_API_KEY"))
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument(
        "--provider",
        choices=["deepseek", "claude", "gemini", "openai", "groq", "openrouter", "ollama"],
        default=os.environ.get("LLM_PROVIDER", "deepseek"),
    )
    parser.add_argument("--provider-key", default=None)
    parser.add_argument("--provider-model", default=None)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--caption-style", default="modern-box")
    parser.add_argument("--num-clips", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    if args.provider_key is None:
        args.provider_key = os.environ.get({
            "deepseek": "DEEPSEEK_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }.get(args.provider, ""), "")

    result = run_pipeline(
        args.source,
        transcription=args.transcription,
        transcription_api_key=args.deepgram_key,
        whisper_model=args.whisper_model,
        provider=args.provider,
        provider_api_key=args.provider_key,
        provider_model=args.provider_model,
        ollama_host=args.ollama_host,
        workdir=args.workdir,
        caption_style=args.caption_style,
        num_clips=args.num_clips,
        render_clips=not args.no_render,
    )
    print(json.dumps({
        "candidates": result["candidates"],
        "clips": result["clips"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
