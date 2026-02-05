#!/usr/bin/env python3
"""
Interactive Voice Chatbot with Feishu (Chrome) Integration
- ASR: OpenAI Whisper (captures Feishu audio from feishu_output.monitor)
- LLM: Local Gateway (OpenAI-compatible)
- TTS: OpenAI TTS (outputs to tts_output virtual sink -> Feishu microphone)
- Audio I/O: PyAudio with PulseAudio virtual devices (no physical sound card needed)

Audio Routing (all virtual):
  对方说话 → 飞书扬声器(feishu_output) → .monitor → ASR → LLM → TTS
                                                                   ↓
                                                              tts_output → tts_mic → 飞书麦克风 → 对方听到

Chrome Feishu Settings:
  Speaker:    Feishu_Speaker
  Microphone: TTS_Microphone
"""

import io
import os
import sys
import wave
import tempfile
import subprocess
from pathlib import Path

import pyaudio
from openai import OpenAI

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# =============================================================================
# Configuration
# =============================================================================
GATEWAY_URL = "http://localhost:18789/v1"
GATEWAY_TOKEN = ""
OPENAI_API_KEY = ""

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 300
SILENCE_DURATION = 1.5
MAX_RECORD_SECONDS = 30

# PulseAudio virtual device names
PA_TTS_SINK = "tts_output"          # TTS outputs here
PA_TTS_MIC = "tts_mic"              # Feishu reads this as microphone
PA_FEISHU_SINK = "feishu_output"    # Feishu outputs audio here (replaces real speaker)

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
gateway_client = OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_TOKEN)


# =============================================================================
# PulseAudio Helpers
# =============================================================================

# Global reference to keep the ALSA error handler callback alive
_alsa_error_handler = None

def suppress_alsa_errors():
    """Suppress ALSA error messages."""
    global _alsa_error_handler
    try:
        from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
        ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        def py_error_handler(filename, line, function, err, fmt):
            pass
        _alsa_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(_alsa_error_handler)
    except:
        pass


def run_pactl(*args) -> str:
    """Run a pactl command and return stdout."""
    try:
        result = subprocess.run(
            ['pactl'] + list(args),
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"⚠ pactl error: {e}")
        return ""


def get_pa_sinks():
    """Get PulseAudio sinks (output devices)."""
    sinks = []
    output = run_pactl('list', 'sinks', 'short')
    for line in output.split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2:
            sinks.append({'index': parts[0], 'name': parts[1]})
    return sinks


def get_pa_sources():
    """Get PulseAudio sources (input devices)."""
    sources = []
    output = run_pactl('list', 'sources', 'short')
    for line in output.split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2:
            sources.append({'index': parts[0], 'name': parts[1]})
    return sources


def pa_device_exists(name, device_type='sink'):
    """Check if a PulseAudio sink or source exists."""
    if device_type == 'sink':
        return any(s['name'] == name for s in get_pa_sinks())
    else:
        return any(s['name'] == name for s in get_pa_sources())


def setup_virtual_devices():
    """Create all PulseAudio virtual devices needed for Feishu integration.

    Creates:
      - feishu_output: Feishu's speaker (Chrome outputs here)
      - tts_output:    TTS audio goes here
      - tts_mic:       Remap of tts_output.monitor so Chrome/Feishu can see it as a mic
    """
    created = []

    # 1. Feishu speaker sink — Chrome Feishu outputs audio here
    if not pa_device_exists(PA_FEISHU_SINK):
        run_pactl('load-module', 'module-null-sink',
                  f'sink_name={PA_FEISHU_SINK}',
                  'sink_properties=device.description="Feishu_Speaker"')
        created.append(PA_FEISHU_SINK)

    # 2. TTS output sink — Python TTS writes audio here
    if not pa_device_exists(PA_TTS_SINK):
        run_pactl('load-module', 'module-null-sink',
                  f'sink_name={PA_TTS_SINK}',
                  'sink_properties=device.description="TTS_Virtual_Speaker"')
        created.append(PA_TTS_SINK)

    # 3. TTS mic (remap source) — Feishu reads this as microphone input
    if not pa_device_exists(PA_TTS_MIC, 'source'):
        run_pactl('load-module', 'module-remap-source',
                  f'master={PA_TTS_SINK}.monitor',
                  f'source_name={PA_TTS_MIC}',
                  'source_properties=device.description="TTS_Microphone"')
        created.append(PA_TTS_MIC)

    if created:
        print(f"✓ Created virtual devices: {', '.join(created)}")
    else:
        print("✓ Virtual devices already exist")

    # Verify
    feishu_monitor = f"{PA_FEISHU_SINK}.monitor"
    if pa_device_exists(feishu_monitor, 'source'):
        print(f"✓ ASR source ready: {feishu_monitor}")
    else:
        print(f"⚠ ASR source '{feishu_monitor}' not found")

    if pa_device_exists(PA_TTS_MIC, 'source'):
        print(f"✓ Feishu mic ready: {PA_TTS_MIC}")
    else:
        print(f"⚠ Feishu mic '{PA_TTS_MIC}' not found")

    print()
    print("📋 Chrome Feishu audio settings:")
    print("   Speaker    → Feishu_Speaker")
    print("   Microphone → TTS_Microphone")


def find_pyaudio_device(pa: pyaudio.PyAudio, pulse_name: str, is_input: bool) -> int | None:
    """Find PyAudio device index matching a PulseAudio device name."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if is_input and info['maxInputChannels'] > 0:
            if pulse_name in info['name'] or info['name'] in pulse_name:
                return i
        elif not is_input and info['maxOutputChannels'] > 0:
            if pulse_name in info['name'] or info['name'] in pulse_name:
                return i
    return None


# =============================================================================
# Main Chatbot
# =============================================================================

class VoiceChatbot:
    def __init__(self, input_device=None, output_device=None,
                 asr_source=None, tts_sink=None):
        suppress_alsa_errors()
        self.audio = pyaudio.PyAudio()
        self.conversation_history = []
        self.is_running = True

        self.input_device = input_device
        self.output_device = output_device

        # Resolve PulseAudio names to PyAudio device indices
        if asr_source:
            idx = find_pyaudio_device(self.audio, asr_source, is_input=True)
            if idx is not None:
                self.input_device = idx
                print(f"✓ ASR input: [{idx}] {self.audio.get_device_info_by_index(idx)['name']}")
            else:
                print(f"⚠ Cannot find PyAudio device for '{asr_source}', using default")

        if tts_sink:
            idx = find_pyaudio_device(self.audio, tts_sink, is_input=False)
            if idx is not None:
                self.output_device = idx
                print(f"✓ TTS output: [{idx}] {self.audio.get_device_info_by_index(idx)['name']}")
            else:
                print(f"⚠ Cannot find PyAudio device for '{tts_sink}', using default")

    def record_audio(self) -> bytes:
        """Record audio until silence is detected.

        Captures audio from feishu_output.monitor (= what Feishu is playing,
        i.e. the other person speaking in the meeting).
        """
        try:
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=CHUNK_SIZE,
            )
        except Exception as e:
            print(f"Error opening audio input: {e}")
            print("Try: python voice_chatbot.py --list-devices")
            return b''

        print("🎤 Listening... (waiting for speech, Ctrl+C to skip)")
        frames = []
        silent_chunks = 0
        has_speech = False
        silence_chunks_threshold = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        max_chunks = int(MAX_RECORD_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
        chunk_count = 0

        try:
            while self.is_running and chunk_count < max_chunks:
                try:
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except Exception as e:
                    print(f"Read error: {e}")
                    break

                frames.append(data)
                chunk_count += 1

                try:
                    samples = [int.from_bytes(data[i:i+2], 'little', signed=True)
                               for i in range(0, len(data), 2)]
                    audio_level = sum(abs(s) for s in samples) // len(samples)
                except:
                    audio_level = 0

                level_bar = "█" * min(int(audio_level / 100), 30)
                print(f"\r  Level: {audio_level:5d} {level_bar:<30}", end="", flush=True)

                if audio_level > SILENCE_THRESHOLD:
                    has_speech = True
                    silent_chunks = 0
                else:
                    silent_chunks += 1

                if has_speech and silent_chunks > silence_chunks_threshold:
                    print("\n✓ Speech detected, processing...")
                    break

        except KeyboardInterrupt:
            print("\n(Skipped)")
            return b''
        finally:
            stream.stop_stream()
            stream.close()

        if not frames:
            return b''

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))

        return wav_buffer.getvalue()

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio using OpenAI Whisper."""
        if not audio_bytes:
            return ""

        print("📝 Transcribing...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )
            return transcript.strip()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def chat(self, user_message: str) -> str:
        """Send message to gateway LLM and get response."""
        print("🤖 Thinking...")

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        response = gateway_client.chat.completions.create(
            model="openclaw:main",
            messages=self.conversation_history,
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })

        return assistant_message

    def speak(self, text: str):
        """Convert text to speech and output to tts_output virtual sink.

        tts_output -> tts_mic -> Feishu picks it up as microphone -> other party hears it.
        """
        print("🔊 Speaking (→ Feishu)...")

        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="pcm",
        )

        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
                output_device_index=self.output_device,
            )
        except Exception as e:
            print(f"Error opening audio output: {e}")
            Path("response.pcm").write_bytes(response.content)
            print("Saved to response.pcm")
            return

        try:
            for chunk in response.iter_bytes(chunk_size=4096):
                if chunk:
                    stream.write(chunk)
        finally:
            stream.stop_stream()
            stream.close()

    def run(self):
        """Main conversation loop."""
        print()
        print("=" * 60)
        print("🎙️  Voice Chatbot (Feishu Integration)")
        print("=" * 60)
        print("Audio routing:")
        print("  ASR  ← feishu_output.monitor (captures meeting audio)")
        print("  TTS  → tts_output → tts_mic → Feishu microphone")
        print("Press Ctrl+C to exit")
        print()

        try:
            while self.is_running:
                audio_data = self.record_audio()
                user_text = self.transcribe(audio_data)

                if not user_text:
                    print("(No speech detected, try again)\n")
                    continue

                print(f"👤 They said: {user_text}")

                if user_text.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print("👋 Goodbye!")
                    self.speak("Goodbye!")
                    break

                response_text = self.chat(user_text)
                print(f"🤖 Reply: {response_text}")

                self.speak(response_text)
                print()

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
        finally:
            self.cleanup()

    def cleanup(self):
        self.is_running = False
        self.audio.terminate()


# =============================================================================
# CLI
# =============================================================================

def list_all_devices():
    """List all audio devices."""
    suppress_alsa_errors()
    p = pyaudio.PyAudio()

    print("\n" + "=" * 60)
    print("PyAudio Devices")
    print("=" * 60)

    print("\nInput (capture):")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} ({info['maxInputChannels']}ch)")

    print("\nOutput (playback):")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']} ({info['maxOutputChannels']}ch)")

    p.terminate()

    print("\n" + "=" * 60)
    print("PulseAudio Sinks")
    print("=" * 60)
    for s in get_pa_sinks():
        print(f"  [{s['index']}] {s['name']}")

    print("\nPulseAudio Sources")
    print("-" * 60)
    for s in get_pa_sources():
        print(f"  [{s['index']}] {s['name']}")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Voice Chatbot with Feishu/PulseAudio Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all devices
  %(prog)s --list-devices

  # Feishu mode - auto setup (recommended)
  %(prog)s --feishu

  # Manual device selection by PyAudio index
  %(prog)s --input-device 5 --output-device 8

  # Manual PulseAudio source/sink names
  %(prog)s --asr-source feishu_output.monitor --tts-sink tts_output

Chrome Feishu settings:
  Speaker    → Feishu_Speaker
  Microphone → TTS_Microphone
        """
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="List all audio devices and exit")
    parser.add_argument("--feishu", action="store_true",
                        help="Auto-setup for Feishu in Chrome: create virtual devices, "
                             "ASR from feishu_output.monitor, TTS to tts_output")
    parser.add_argument("--input-device", type=int, default=None,
                        help="PyAudio input device index (for ASR)")
    parser.add_argument("--output-device", type=int, default=None,
                        help="PyAudio output device index (for TTS)")
    parser.add_argument("--asr-source", type=str, default=None,
                        help="PulseAudio source name for ASR capture")
    parser.add_argument("--tts-sink", type=str, default=None,
                        help="PulseAudio sink name for TTS output")
    args = parser.parse_args()

    suppress_alsa_errors()

    if args.list_devices:
        list_all_devices()
        return

    asr_source = args.asr_source
    tts_sink = args.tts_sink

    if args.feishu:
        print("🔧 Setting up Feishu integration...\n")
        setup_virtual_devices()
        asr_source = asr_source or f"{PA_FEISHU_SINK}.monitor"
        tts_sink = tts_sink or PA_TTS_SINK
        print()

    chatbot = VoiceChatbot(
        input_device=args.input_device,
        output_device=args.output_device,
        asr_source=asr_source,
        tts_sink=tts_sink,
    )
    chatbot.run()


if __name__ == "__main__":
    main()
