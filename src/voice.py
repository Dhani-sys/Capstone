"""Step 4: voice layer, Gnani STT/TTS only (no other speech service).

Setup (from the brief):
    1. Create a Gnani account and API key at app.gnani.ai/voice.
    2. ``export GNANI_API_KEY=...`` -- never put the key in code or git.
    3. ``pip install gnani-vachana``

The brain is unchanged: audio -> Gnani STT -> Step 3 agent -> Gnani TTS.
STT clips are limited to 60 seconds (Gnani STT REST docs). TTS always
names its model: the SDK's older default is no longer served, so this code
uses ``timbre-v2.5`` (voices include Kaveri, Trupti, Pranav, Shlok) unless
``--model`` overrides it. Replies are written for the ear: dates with the
month name, order IDs in groups of four digits (STY-3).
"""
from __future__ import annotations

import os
import re


def _voice_ready_text(text: str) -> str:
    """Make a reply speakable: break order IDs into groups of four digits."""
    text = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", text)
    return text


class GnaniVoice:
    def __init__(self, language: str = "en-IN", voice: str = "Kaveri",
                 model: str = "timbre-v2.5"):
        if not os.environ.get("GNANI_API_KEY"):
            raise RuntimeError(
                "GNANI_API_KEY is not set. Create a key at app.gnani.ai/voice "
                "and export GNANI_API_KEY before using voice.")
        try:
            from gnani.stt import GnaniSTTClient
            from gnani.tts import GnaniTTSClient
        except ImportError as exc:
            raise RuntimeError(
                "gnani-vachana is not installed. Run: pip install gnani-vachana"
            ) from exc
        self.stt = GnaniSTTClient()  # reads GNANI_API_KEY from the environment
        self.tts = GnaniTTSClient()
        self.language = language
        self.voice = voice
        self.model = model

    def transcribe(self, audio_path: str) -> str:
        result = self.stt.transcribe(audio_path, language_code=self.language)
        if isinstance(result, dict):
            return (result.get("transcript") or "").strip()
        return str(result).strip()

    def synthesize(self, text: str, output_file: str) -> str:
        self.tts.synthesize(
            _voice_ready_text(text), voice=self.voice, model=self.model,
            language=self.language, output_file=output_file)
        return output_file

    def voice_turn(self, audio_path: str, agent) -> tuple:
        """One spoken turn: STT -> agent -> TTS. Returns (heard, reply)."""
        heard = self.transcribe(audio_path)
        if not heard:
            reply = "Sorry, I didn't catch that. Could you say it again?"
        else:
            reply = agent.answer_single(heard).text
        return heard, reply
