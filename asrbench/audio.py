import io
from pydub import AudioSegment


def load_audio(path: str) -> tuple[bytes, str, float]:
    """Load audio file and return (raw_bytes, format_str, duration_ms)."""
    with open(path, "rb") as f:
        data = f.read()
    fmt = "pcm"
    if path.endswith(".mp3"):
        fmt = "mp3"
    elif path.endswith(".wav"):
        fmt = "wav"
    duration_ms = get_audio_duration_ms(data, fmt)
    return data, fmt, duration_ms


def get_audio_duration_ms(audio_data: bytes, audio_format: str) -> float:
    """Get audio duration in milliseconds."""
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format)
        return len(seg)
    except Exception:
        return 0.0


def convert_to_pcm(audio_data: bytes, src_format: str, target_sample_rate: int = 16000) -> bytes:
    """Convert audio bytes to 16-bit mono PCM at the target sample rate."""
    seg = AudioSegment.from_file(io.BytesIO(audio_data), format=src_format)
    seg = seg.set_frame_rate(target_sample_rate).set_channels(1).set_sample_width(2)
    return seg.raw_data


def ensure_format(audio_data: bytes, src_format: str, supported_formats: frozenset[str],
                  sample_rate: int = 16000) -> tuple[bytes, str]:
    """Return audio in a format the provider supports, converting if needed."""
    if src_format in supported_formats:
        return audio_data, src_format
    pcm_data = convert_to_pcm(audio_data, src_format, sample_rate)
    if "pcm" in supported_formats:
        return pcm_data, "pcm"
    if "wav" in supported_formats:
        return _pcm_to_wav(pcm_data, sample_rate), "wav"
    return pcm_data, "pcm"


def _pcm_to_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()
