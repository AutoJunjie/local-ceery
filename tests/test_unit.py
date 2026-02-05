"""
Unit tests for voice chatbot helper functions.

Run with: pytest tests/test_unit.py -v
"""

import io
import pytest
from unittest.mock import patch, MagicMock
import subprocess

# conftest.py handles mocking of pyaudio and openai before this import
import local_ceery as lc


class TestPactlHelpers:
    """Tests for PulseAudio helper functions."""

    def test_run_pactl_success(self):
        """Test successful pactl command execution."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout="output\n", returncode=0)
            result = lc.run_pactl('list', 'sinks')
            assert result == "output"
            mock_run.assert_called_once()

    def test_run_pactl_timeout(self):
        """Test pactl command timeout handling."""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('pactl', 5)):
            result = lc.run_pactl('list', 'sinks')
            assert result == ""

    def test_run_pactl_error(self):
        """Test pactl command error handling."""
        with patch('subprocess.run', side_effect=Exception("command not found")):
            result = lc.run_pactl('invalid')
            assert result == ""

    def test_get_pa_sinks(self):
        """Test parsing PulseAudio sinks."""
        with patch.object(lc, 'run_pactl', return_value="0\tsink1\tmodule\n1\tsink2\tmodule"):
            sinks = lc.get_pa_sinks()
            assert len(sinks) == 2
            assert sinks[0] == {'index': '0', 'name': 'sink1'}
            assert sinks[1] == {'index': '1', 'name': 'sink2'}

    def test_get_pa_sinks_empty(self):
        """Test parsing empty sinks list."""
        with patch.object(lc, 'run_pactl', return_value=""):
            sinks = lc.get_pa_sinks()
            assert sinks == []

    def test_get_pa_sources(self):
        """Test parsing PulseAudio sources."""
        with patch.object(lc, 'run_pactl', return_value="0\tsource1\tmodule\n1\tsource2\tmodule"):
            sources = lc.get_pa_sources()
            assert len(sources) == 2
            assert sources[0] == {'index': '0', 'name': 'source1'}

    def test_pa_device_exists_sink_found(self):
        """Test sink existence check - found."""
        with patch.object(lc, 'get_pa_sinks', return_value=[{'index': '0', 'name': 'test_sink'}]):
            assert lc.pa_device_exists('test_sink', 'sink') is True

    def test_pa_device_exists_sink_not_found(self):
        """Test sink existence check - not found."""
        with patch.object(lc, 'get_pa_sinks', return_value=[{'index': '0', 'name': 'other_sink'}]):
            assert lc.pa_device_exists('test_sink', 'sink') is False

    def test_pa_device_exists_source_found(self):
        """Test source existence check - found."""
        with patch.object(lc, 'get_pa_sources', return_value=[{'index': '0', 'name': 'test_source'}]):
            assert lc.pa_device_exists('test_source', 'source') is True


class TestFindPyaudioDevice:
    """Tests for PyAudio device lookup."""

    def test_find_input_device(self, mock_pyaudio):
        """Test finding input device by PulseAudio name."""
        pa = mock_pyaudio.return_value
        result = lc.find_pyaudio_device(pa, 'feishu_output.monitor', is_input=True)
        assert result == 1

    def test_find_output_device(self, mock_pyaudio):
        """Test finding output device by PulseAudio name."""
        pa = mock_pyaudio.return_value
        result = lc.find_pyaudio_device(pa, 'tts_output', is_input=False)
        assert result == 2

    def test_find_device_not_found(self, mock_pyaudio):
        """Test device not found returns None."""
        pa = mock_pyaudio.return_value
        result = lc.find_pyaudio_device(pa, 'nonexistent_device', is_input=True)
        assert result is None

    def test_find_device_partial_match(self, mock_pyaudio):
        """Test partial name matching."""
        pa = mock_pyaudio.return_value
        result = lc.find_pyaudio_device(pa, 'feishu', is_input=True)
        assert result == 1  # Should match 'feishu_output.monitor'


class TestSuppressAlsaErrors:
    """Tests for ALSA error suppression."""

    def test_suppress_alsa_errors_success(self):
        """Test ALSA error suppression loads successfully."""
        with patch('ctypes.cdll.LoadLibrary') as mock_load:
            mock_lib = MagicMock()
            mock_load.return_value = mock_lib
            lc.suppress_alsa_errors()
            # Should not raise

    def test_suppress_alsa_errors_missing_library(self):
        """Test graceful handling when libasound is not available."""
        with patch('ctypes.cdll.LoadLibrary', side_effect=OSError("library not found")):
            # Should not raise, just silently fail
            lc.suppress_alsa_errors()


class TestVoiceChatbotInit:
    """Tests for VoiceChatbot initialization."""

    def test_init_default_devices(self, mock_pyaudio):
        """Test initialization with default devices."""
        with patch.object(lc, 'suppress_alsa_errors'):
            chatbot = lc.VoiceChatbot()
            assert chatbot.input_device is None
            assert chatbot.output_device is None
            assert chatbot.conversation_history == []
            assert chatbot.is_running is True
            chatbot.cleanup()

    def test_init_with_asr_source(self, mock_pyaudio):
        """Test initialization with ASR source name."""
        with patch.object(lc, 'suppress_alsa_errors'):
            chatbot = lc.VoiceChatbot(asr_source='feishu_output.monitor')
            assert chatbot.input_device == 1
            chatbot.cleanup()

    def test_init_with_tts_sink(self, mock_pyaudio):
        """Test initialization with TTS sink name."""
        with patch.object(lc, 'suppress_alsa_errors'):
            chatbot = lc.VoiceChatbot(tts_sink='tts_output')
            assert chatbot.output_device == 2
            chatbot.cleanup()

    def test_init_device_not_found(self, mock_pyaudio, capsys):
        """Test initialization when device is not found."""
        with patch.object(lc, 'suppress_alsa_errors'):
            chatbot = lc.VoiceChatbot(asr_source='nonexistent')
            assert chatbot.input_device is None  # Falls back to default
            captured = capsys.readouterr()
            assert "Cannot find PyAudio device" in captured.out
            chatbot.cleanup()


class TestTranscribe:
    """Tests for transcribe method."""

    def test_transcribe_empty_audio(self, mock_pyaudio):
        """Test transcription with empty audio returns empty string."""
        with patch.object(lc, 'suppress_alsa_errors'):
            chatbot = lc.VoiceChatbot()
            result = chatbot.transcribe(b'')
            assert result == ""
            chatbot.cleanup()

    def test_transcribe_uses_bytesio(self, mock_pyaudio, sample_audio_bytes):
        """Test that transcribe uses BytesIO (not temp file)."""
        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.openai_client.audio.transcriptions, 'create') as mock_create:
            mock_create.return_value = "test transcription"

            chatbot = lc.VoiceChatbot()
            result = chatbot.transcribe(sample_audio_bytes)

            # Verify BytesIO was passed (not a file path)
            call_args = mock_create.call_args
            file_arg = call_args.kwargs.get('file') or call_args[1].get('file')
            assert isinstance(file_arg, io.BytesIO)
            assert file_arg.name == "audio.wav"
            assert result == "test transcription"
            chatbot.cleanup()


class TestChat:
    """Tests for chat method."""

    def test_chat_appends_history(self, mock_pyaudio):
        """Test that chat appends messages to conversation history."""
        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.gateway_client.chat.completions, 'create') as mock_create:

            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Bot response"
            mock_create.return_value = response

            chatbot = lc.VoiceChatbot()
            result = chatbot.chat("Hello")

            assert len(chatbot.conversation_history) == 2
            assert chatbot.conversation_history[0] == {"role": "user", "content": "Hello"}
            assert chatbot.conversation_history[1] == {"role": "assistant", "content": "Bot response"}
            assert result == "Bot response"
            chatbot.cleanup()


class TestSetupVirtualDevices:
    """Tests for virtual device setup."""

    def test_setup_creates_missing_devices(self, capsys):
        """Test that setup creates missing virtual devices."""
        with patch.object(lc, 'pa_device_exists', return_value=False), \
             patch.object(lc, 'run_pactl') as mock_pactl:

            lc.setup_virtual_devices()

            # Should call load-module for each device
            assert mock_pactl.call_count >= 3
            captured = capsys.readouterr()
            assert "Created virtual devices" in captured.out

    def test_setup_skips_existing_devices(self, capsys):
        """Test that setup skips already existing devices."""
        with patch.object(lc, 'pa_device_exists', return_value=True), \
             patch.object(lc, 'run_pactl') as mock_pactl:

            lc.setup_virtual_devices()

            captured = capsys.readouterr()
            assert "already exist" in captured.out
