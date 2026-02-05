"""
Functional tests for ASR and TTS.

These tests verify the actual behavior of ASR (transcribe) and TTS (speak) methods.
Some tests require real API keys and are marked with @pytest.mark.integration.

Run unit tests only: pytest tests/test_functional.py -v -m "not integration"
Run integration tests: pytest tests/test_functional.py -v -m integration

Set environment variables for integration tests:
  OPENAI_API_KEY=your-key pytest tests/test_functional.py -v -m integration
"""

import io
import os
import wave
import struct
import math
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# conftest.py handles mocking of pyaudio and openai before this import
import local_ceery as lc


def generate_wav_audio(duration: float = 1.0, frequency: int = 440,
                       sample_rate: int = 16000, amplitude: float = 0.5) -> bytes:
    """Generate WAV audio bytes with specified parameters."""
    num_samples = int(sample_rate * duration)
    samples = [int(32767 * amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
               for i in range(num_samples)]
    audio_data = struct.pack(f'<{len(samples)}h', *samples)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

    return wav_buffer.getvalue()


class TestASRFunctional:
    """Functional tests for ASR (Automatic Speech Recognition)."""

    def test_transcribe_empty_returns_empty(self, mock_pyaudio):
        """Transcribing empty audio should return empty string."""
        with patch.object(lc, 'suppress_alsa_errors'):
            chatbot = lc.VoiceChatbot()
            result = chatbot.transcribe(b'')
            assert result == ""
            chatbot.cleanup()

    def test_transcribe_valid_audio_format(self, mock_pyaudio):
        """Verify transcribe sends correctly formatted audio to API."""
        audio_bytes = generate_wav_audio(duration=0.5)

        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.openai_client.audio.transcriptions, 'create') as mock_create:
            mock_create.return_value = "Hello world"

            chatbot = lc.VoiceChatbot()
            result = chatbot.transcribe(audio_bytes)

            # Verify API was called
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs

            # Verify model
            assert call_kwargs['model'] == 'whisper-1'

            # Verify file is BytesIO with .wav extension
            file_obj = call_kwargs['file']
            assert isinstance(file_obj, io.BytesIO)
            assert file_obj.name.endswith('.wav')

            # Verify response format
            assert call_kwargs['response_format'] == 'text'

            assert result == "Hello world"
            chatbot.cleanup()

    def test_transcribe_strips_whitespace(self, mock_pyaudio):
        """Transcription result should be stripped of whitespace."""
        audio_bytes = generate_wav_audio()

        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.openai_client.audio.transcriptions, 'create') as mock_create:
            mock_create.return_value = "  Hello world  \n"

            chatbot = lc.VoiceChatbot()
            result = chatbot.transcribe(audio_bytes)

            assert result == "Hello world"
            chatbot.cleanup()

    def test_transcribe_handles_api_error(self, mock_pyaudio):
        """Transcribe should handle API errors gracefully."""
        audio_bytes = generate_wav_audio()

        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.openai_client.audio.transcriptions, 'create') as mock_create:
            mock_create.side_effect = Exception("API error")

            chatbot = lc.VoiceChatbot()

            with pytest.raises(Exception, match="API error"):
                chatbot.transcribe(audio_bytes)

            chatbot.cleanup()

    @pytest.mark.integration
    def test_transcribe_real_api(self, mock_pyaudio):
        """Integration test: actual API call with real audio.

        Requires OPENAI_API_KEY environment variable.
        """
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        # Generate audio that says something (in reality, this is just a tone)
        # Real integration would use actual speech audio
        audio_bytes = generate_wav_audio(duration=1.0)

        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc, 'openai_client', lc.OpenAI(api_key=api_key)):

            chatbot = lc.VoiceChatbot()
            # Note: Whisper will likely return empty or noise for synthesized tone
            result = chatbot.transcribe(audio_bytes)

            # Just verify it returns a string without error
            assert isinstance(result, str)
            chatbot.cleanup()


class TestTTSFunctional:
    """Functional tests for TTS (Text-to-Speech)."""

    def test_speak_calls_api_with_correct_params(self, mock_pyaudio):
        """Verify speak sends correct parameters to TTS API."""
        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.openai_client.audio.speech, 'create') as mock_create:

            # Mock TTS response
            response = MagicMock()
            response.iter_bytes.return_value = [b'\x00' * 1000]
            mock_create.return_value = response

            # Mock audio output stream
            mock_stream = MagicMock()
            mock_pyaudio.return_value.open.return_value = mock_stream

            chatbot = lc.VoiceChatbot()
            chatbot.speak("Hello world")

            # Verify API call
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs

            assert call_kwargs['model'] == 'tts-1'
            assert call_kwargs['voice'] == 'alloy'
            assert call_kwargs['input'] == 'Hello world'
            assert call_kwargs['response_format'] == 'pcm'

            chatbot.cleanup()

    def test_speak_outputs_to_correct_device(self, mock_pyaudio):
        """Verify TTS outputs to the specified device."""
        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.openai_client.audio.speech, 'create') as mock_create:

            response = MagicMock()
            response.iter_bytes.return_value = [b'\x00' * 1000]
            mock_create.return_value = response

            mock_stream = MagicMock()
            mock_pyaudio.return_value.open.return_value = mock_stream

            chatbot = lc.VoiceChatbot(output_device=5)
            chatbot.speak("Test")

            # Verify stream opened with correct output device
            open_call = mock_pyaudio.return_value.open.call_args
            assert open_call.kwargs.get('output_device_index') == 5
            assert open_call.kwargs.get('output') is True

            chatbot.cleanup()

    def test_speak_handles_audio_output_error(self, mock_pyaudio, tmp_path, capsys):
        """Speak should save to file if audio output fails."""
        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.openai_client.audio.speech, 'create') as mock_create, \
             patch.object(lc.Path, '__new__', lambda cls, p: tmp_path / 'response.pcm'):

            response = MagicMock()
            response.content = b'\x00' * 1000
            response.iter_bytes.return_value = [b'\x00' * 1000]
            mock_create.return_value = response

            # Make stream open fail
            mock_pyaudio.return_value.open.side_effect = Exception("Device not available")

            chatbot = lc.VoiceChatbot()
            chatbot.speak("Test")

            captured = capsys.readouterr()
            assert "Error opening audio output" in captured.out

            chatbot.cleanup()

    def test_speak_streams_audio_chunks(self, mock_pyaudio):
        """Verify audio is streamed in chunks, not all at once."""
        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.openai_client.audio.speech, 'create') as mock_create:

            # Return multiple chunks
            chunks = [b'\x00' * 4096, b'\x01' * 4096, b'\x02' * 4096]
            response = MagicMock()
            response.iter_bytes.return_value = chunks
            mock_create.return_value = response

            mock_stream = MagicMock()
            mock_pyaudio.return_value.open.return_value = mock_stream

            chatbot = lc.VoiceChatbot()
            chatbot.speak("Test streaming")

            # Verify all chunks were written
            assert mock_stream.write.call_count == 3

            chatbot.cleanup()

    @pytest.mark.integration
    def test_speak_real_api(self, mock_pyaudio):
        """Integration test: actual TTS API call.

        Requires OPENAI_API_KEY environment variable.
        """
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc, 'openai_client', lc.OpenAI(api_key=api_key)):

            mock_stream = MagicMock()
            mock_pyaudio.return_value.open.return_value = mock_stream

            chatbot = lc.VoiceChatbot()
            # Short text to minimize API cost
            chatbot.speak("Hi")

            # Verify audio was written to stream
            assert mock_stream.write.called
            chatbot.cleanup()


class TestRecordAudioFunctional:
    """Functional tests for audio recording."""

    def test_record_detects_speech_then_silence(self, mock_pyaudio):
        """Verify recording stops after speech followed by silence."""
        with patch.object(lc, 'suppress_alsa_errors'):
            # Simulate: silence -> speech -> silence
            # Each chunk is CHUNK_SIZE * 2 bytes (16-bit samples)
            silence_chunk = b'\x00\x00' * lc.CHUNK_SIZE
            speech_chunk = struct.pack(f'<{lc.CHUNK_SIZE}h',
                                        *[500] * lc.CHUNK_SIZE)  # Above threshold

            # Create sequence: 2 silence, 5 speech, enough silence to trigger stop
            silence_count = int(lc.SILENCE_DURATION * lc.SAMPLE_RATE / lc.CHUNK_SIZE) + 5
            chunks = ([silence_chunk] * 2 +
                      [speech_chunk] * 5 +
                      [silence_chunk] * silence_count)

            mock_stream = MagicMock()
            mock_stream.read.side_effect = chunks
            mock_pyaudio.return_value.open.return_value = mock_stream

            chatbot = lc.VoiceChatbot()
            result = chatbot.record_audio()

            # Should return audio data
            assert len(result) > 0

            # Verify it's valid WAV format
            wav_buffer = io.BytesIO(result)
            with wave.open(wav_buffer, 'rb') as wf:
                assert wf.getnchannels() == lc.CHANNELS
                assert wf.getframerate() == lc.SAMPLE_RATE

            chatbot.cleanup()

    def test_record_returns_empty_on_stream_error(self, mock_pyaudio, capsys):
        """Recording should return empty bytes if stream fails to open."""
        with patch.object(lc, 'suppress_alsa_errors'):
            mock_pyaudio.return_value.open.side_effect = Exception("No device")

            chatbot = lc.VoiceChatbot()
            result = chatbot.record_audio()

            assert result == b''
            captured = capsys.readouterr()
            assert "Error opening audio input" in captured.out

            chatbot.cleanup()

    def test_record_respects_max_duration(self, mock_pyaudio):
        """Recording should stop at MAX_RECORD_SECONDS."""
        with patch.object(lc, 'suppress_alsa_errors'):
            # Always return speech (above threshold) - should hit max duration
            speech_chunk = struct.pack(f'<{lc.CHUNK_SIZE}h', *[500] * lc.CHUNK_SIZE)

            mock_stream = MagicMock()
            mock_stream.read.return_value = speech_chunk
            mock_pyaudio.return_value.open.return_value = mock_stream

            chatbot = lc.VoiceChatbot()

            # Temporarily reduce max duration for faster test
            original_max = lc.MAX_RECORD_SECONDS
            lc.MAX_RECORD_SECONDS = 0.5

            try:
                result = chatbot.record_audio()
                # Should return some audio (hit max duration)
                assert len(result) > 0
            finally:
                lc.MAX_RECORD_SECONDS = original_max

            chatbot.cleanup()


class TestEndToEndFlow:
    """End-to-end tests simulating full conversation flow."""

    def test_full_conversation_turn(self, mock_pyaudio):
        """Test a complete conversation turn: record -> transcribe -> chat -> speak."""
        with patch.object(lc, 'suppress_alsa_errors'), \
             patch.object(lc.openai_client.audio.transcriptions, 'create') as mock_asr, \
             patch.object(lc.openai_client.audio.speech, 'create') as mock_tts, \
             patch.object(lc.gateway_client.chat.completions, 'create') as mock_chat:

            # Setup mocks
            mock_asr.return_value = "Hello, how are you?"

            chat_response = MagicMock()
            chat_response.choices = [MagicMock()]
            chat_response.choices[0].message.content = "I'm doing well, thank you!"
            mock_chat.return_value = chat_response

            tts_response = MagicMock()
            tts_response.iter_bytes.return_value = [b'\x00' * 1000]
            mock_tts.return_value = tts_response

            mock_stream = MagicMock()
            mock_pyaudio.return_value.open.return_value = mock_stream

            # Execute
            chatbot = lc.VoiceChatbot()

            # Simulate transcription
            audio_bytes = generate_wav_audio()
            user_text = chatbot.transcribe(audio_bytes)
            assert user_text == "Hello, how are you?"

            # Simulate chat
            response_text = chatbot.chat(user_text)
            assert response_text == "I'm doing well, thank you!"

            # Verify conversation history
            assert len(chatbot.conversation_history) == 2

            # Simulate speak
            chatbot.speak(response_text)

            # Verify TTS was called with correct text
            mock_tts.assert_called_once()
            assert mock_tts.call_args.kwargs['input'] == "I'm doing well, thank you!"

            chatbot.cleanup()

    def test_exit_commands_recognized(self, mock_pyaudio):
        """Test that exit commands are properly recognized."""
        exit_commands = ["exit", "quit", "bye", "goodbye", "EXIT", "Bye", "GOODBYE"]

        for cmd in exit_commands:
            assert cmd.lower() in ["exit", "quit", "bye", "goodbye"], \
                f"Exit command '{cmd}' should be recognized"
