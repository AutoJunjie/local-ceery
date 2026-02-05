"""
Pytest fixtures and configuration for voice chatbot tests.

This file handles mocking of external dependencies (pyaudio, openai)
to allow tests to run without actual hardware or API keys.
"""

import io
import os
import sys
import wave
import struct
import pytest
from unittest.mock import MagicMock, patch

# =============================================================================
# Add project root to path
# =============================================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# =============================================================================
# Mock external modules before importing local_ceery
# =============================================================================

# Create mock pyaudio module
mock_pyaudio_module = MagicMock()
mock_pyaudio_module.PyAudio = MagicMock()
mock_pyaudio_module.paInt16 = 8  # Actual pyaudio value
sys.modules['pyaudio'] = mock_pyaudio_module

# Create mock openai module
mock_openai_module = MagicMock()
mock_openai_class = MagicMock()
mock_openai_module.OpenAI = mock_openai_class
sys.modules['openai'] = mock_openai_module

# Now we can safely import local_ceery
import local_ceery as lc


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_pyaudio():
    """Mock PyAudio for tests that don't need real audio devices."""
    instance = MagicMock()
    instance.get_device_count.return_value = 3
    instance.get_device_info_by_index.side_effect = lambda i: {
        0: {'name': 'default', 'maxInputChannels': 2, 'maxOutputChannels': 2},
        1: {'name': 'feishu_output.monitor', 'maxInputChannels': 2, 'maxOutputChannels': 0},
        2: {'name': 'tts_output', 'maxInputChannels': 0, 'maxOutputChannels': 2},
    }.get(i, {'name': 'unknown', 'maxInputChannels': 0, 'maxOutputChannels': 0})
    instance.get_sample_size.return_value = 2
    instance.open.return_value = MagicMock()

    with patch.object(lc.pyaudio, 'PyAudio', return_value=instance):
        yield MagicMock(return_value=instance)


@pytest.fixture
def sample_audio_bytes():
    """Generate sample WAV audio bytes for testing."""
    import math
    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440  # Hz (A4 note)

    num_samples = int(sample_rate * duration)
    samples = [int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
               for i in range(num_samples)]

    # Pack as 16-bit signed integers
    audio_data = struct.pack(f'<{len(samples)}h', *samples)

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

    return wav_buffer.getvalue()


@pytest.fixture
def sample_audio_with_speech():
    """Generate audio that simulates speech (higher amplitude)."""
    import math
    sample_rate = 16000
    duration = 2.0

    num_samples = int(sample_rate * duration)
    samples = []

    for i in range(num_samples):
        # First half: speech (high amplitude)
        if i < num_samples // 2:
            value = int(32767 * 0.8 * math.sin(2 * math.pi * 200 * i / sample_rate))
        # Second half: silence (low amplitude)
        else:
            value = int(32767 * 0.01 * math.sin(2 * math.pi * 200 * i / sample_rate))
        samples.append(value)

    audio_data = struct.pack(f'<{len(samples)}h', *samples)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

    return wav_buffer.getvalue()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for ASR/TTS tests."""
    with patch.object(lc, 'openai_client') as mock:
        # Mock transcription
        mock.audio.transcriptions.create.return_value = "Hello, this is a test."

        # Mock TTS
        tts_response = MagicMock()
        tts_response.content = b'\x00' * 4800  # 0.1s of silence at 24kHz
        tts_response.iter_bytes.return_value = [b'\x00' * 4800]
        mock.audio.speech.create.return_value = tts_response

        yield mock


@pytest.fixture
def mock_gateway_client():
    """Mock gateway LLM client."""
    with patch.object(lc, 'gateway_client') as mock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "This is a test response."
        mock.chat.completions.create.return_value = response
        yield mock


@pytest.fixture
def mock_pactl():
    """Mock pactl subprocess calls."""
    def run_pactl_side_effect(*args):
        if 'sinks' in args:
            return "0\tfeishu_output\tmodule-null-sink\n1\ttts_output\tmodule-null-sink"
        elif 'sources' in args:
            return "0\tfeishu_output.monitor\tmodule-null-sink\n1\ttts_mic\tmodule-remap-source"
        return ""

    with patch.object(lc, 'run_pactl', side_effect=run_pactl_side_effect) as mock:
        yield mock
