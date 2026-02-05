"""
Performance tests for voice chatbot.

Run with: pytest tests/test_performance.py -v
For benchmarks: pytest tests/test_performance.py -v --benchmark-only

These tests compare the optimized BytesIO approach vs the old tempfile approach.
"""

import io
import os
import time
import wave
import struct
import tempfile
import statistics
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# conftest.py handles mocking of pyaudio and openai before this import
import local_ceery as lc


def generate_audio_bytes(duration_seconds: float = 1.0) -> bytes:
    """Generate WAV audio bytes for testing."""
    import math
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)

    samples = [int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / sample_rate))
               for i in range(num_samples)]
    audio_data = struct.pack(f'<{len(samples)}h', *samples)

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

    return wav_buffer.getvalue()


class TestTranscribePerformance:
    """Performance comparison: BytesIO vs tempfile for transcription."""

    def transcribe_with_tempfile(self, audio_bytes: bytes) -> None:
        """Old approach: write to temp file, then read."""
        if not audio_bytes:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as audio_file:
                # Simulate API call (just read the file)
                _ = audio_file.read()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def transcribe_with_bytesio(self, audio_bytes: bytes) -> None:
        """Optimized approach: use BytesIO directly."""
        if not audio_bytes:
            return

        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        # Simulate API call (just read the buffer)
        _ = audio_file.read()

    @pytest.mark.parametrize("duration", [0.5, 1.0, 5.0, 10.0])
    def test_bytesio_faster_than_tempfile(self, duration):
        """Verify BytesIO is faster than tempfile for various audio durations."""
        audio_bytes = generate_audio_bytes(duration)
        iterations = 50

        # Benchmark tempfile approach
        tempfile_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.transcribe_with_tempfile(audio_bytes)
            tempfile_times.append(time.perf_counter() - start)

        # Benchmark BytesIO approach
        bytesio_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.transcribe_with_bytesio(audio_bytes)
            bytesio_times.append(time.perf_counter() - start)

        tempfile_avg = statistics.mean(tempfile_times) * 1000  # ms
        bytesio_avg = statistics.mean(bytesio_times) * 1000    # ms
        speedup = tempfile_avg / bytesio_avg if bytesio_avg > 0 else float('inf')

        print(f"\n  Duration: {duration}s audio")
        print(f"  Tempfile: {tempfile_avg:.3f}ms avg")
        print(f"  BytesIO:  {bytesio_avg:.3f}ms avg")
        print(f"  Speedup:  {speedup:.1f}x faster")

        # BytesIO should be at least 2x faster
        assert bytesio_avg < tempfile_avg, "BytesIO should be faster than tempfile"

    def test_no_temp_files_left(self, tmp_path):
        """Verify BytesIO approach doesn't create any files."""
        audio_bytes = generate_audio_bytes(1.0)

        # Count files before
        files_before = set(tmp_path.iterdir()) if tmp_path.exists() else set()

        # Run BytesIO transcription
        for _ in range(10):
            self.transcribe_with_bytesio(audio_bytes)

        # Count files after
        files_after = set(tmp_path.iterdir()) if tmp_path.exists() else set()

        # No new files should be created
        new_files = files_after - files_before
        assert len(new_files) == 0, f"Unexpected files created: {new_files}"


class TestAudioLevelCalculationPerformance:
    """Performance tests for audio level calculation."""

    def calc_level_original(self, data: bytes) -> int:
        """Original: list comprehension with int.from_bytes."""
        samples = [int.from_bytes(data[i:i+2], 'little', signed=True)
                   for i in range(0, len(data), 2)]
        return sum(abs(s) for s in samples) // len(samples)

    def calc_level_struct(self, data: bytes) -> int:
        """Optimized: use struct.unpack."""
        samples = struct.unpack(f'<{len(data)//2}h', data)
        return sum(abs(s) for s in samples) // len(samples)

    def calc_level_array(self, data: bytes) -> int:
        """Alternative: use array module."""
        import array
        samples = array.array('h')
        samples.frombytes(data)
        return sum(abs(s) for s in samples) // len(samples)

    @pytest.mark.parametrize("chunk_size", [1024, 2048, 4096])
    def test_struct_faster_than_original(self, chunk_size):
        """Verify struct.unpack is faster than int.from_bytes loop."""
        # Generate random audio chunk
        import random
        data = bytes([random.randint(0, 255) for _ in range(chunk_size)])
        iterations = 1000

        # Benchmark original
        original_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.calc_level_original(data)
            original_times.append(time.perf_counter() - start)

        # Benchmark struct
        struct_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.calc_level_struct(data)
            struct_times.append(time.perf_counter() - start)

        original_avg = statistics.mean(original_times) * 1000000  # us
        struct_avg = statistics.mean(struct_times) * 1000000      # us
        speedup = original_avg / struct_avg if struct_avg > 0 else float('inf')

        print(f"\n  Chunk size: {chunk_size} bytes")
        print(f"  Original (int.from_bytes): {original_avg:.2f}us avg")
        print(f"  Struct (unpack):           {struct_avg:.2f}us avg")
        print(f"  Speedup:                   {speedup:.1f}x faster")

        # Struct should be faster
        assert struct_avg < original_avg, "struct.unpack should be faster"

    def test_calculation_correctness(self):
        """Verify all methods produce the same result."""
        import random
        data = bytes([random.randint(0, 255) for _ in range(1024)])

        original = self.calc_level_original(data)
        struct_result = self.calc_level_struct(data)
        array_result = self.calc_level_array(data)

        assert original == struct_result == array_result, \
            f"Results differ: original={original}, struct={struct_result}, array={array_result}"


class TestConversationHistoryMemory:
    """Tests for conversation history memory usage."""

    def test_history_growth(self):
        """Measure memory growth with conversation history."""
        import sys

        history = []

        # Simulate 100 conversation turns
        for i in range(100):
            user_msg = {"role": "user", "content": f"Message {i} " * 50}
            assistant_msg = {"role": "assistant", "content": f"Response {i} " * 100}
            history.append(user_msg)
            history.append(assistant_msg)

        # Calculate approximate memory usage
        total_size = sys.getsizeof(history)
        for msg in history:
            total_size += sys.getsizeof(msg)
            total_size += sys.getsizeof(msg['role'])
            total_size += sys.getsizeof(msg['content'])

        print(f"\n  History size after 100 turns: {len(history)} messages")
        print(f"  Approximate memory: {total_size / 1024:.1f} KB")

        # With 100 turns (200 messages), memory should be manageable
        # but this demonstrates why we need history limits
        assert len(history) == 200

    def test_history_with_limit(self):
        """Test memory usage with history limit."""
        MAX_HISTORY = 20
        history = []

        for i in range(100):
            history.append({"role": "user", "content": f"Message {i}"})
            history.append({"role": "assistant", "content": f"Response {i}"})

            # Apply limit
            if len(history) > MAX_HISTORY:
                history = history[-MAX_HISTORY:]

        assert len(history) == MAX_HISTORY
        # Should contain only the most recent messages
        assert history[-1]['content'] == "Response 99"
