"""Tools to better processing and edition of audio."""

from moviepy.audio.tools.align import (
    AudioAlignmentError,
    AudioOffsetEstimate,
    AudioWindowEstimate,
    align_audio_clip,
    align_audio_to_video,
    estimate_audio_offset,
)
from moviepy.audio.tools.cuts import find_audio_period


__all__ = [
    "AudioAlignmentError",
    "AudioOffsetEstimate",
    "AudioWindowEstimate",
    "align_audio_clip",
    "align_audio_to_video",
    "estimate_audio_offset",
    "find_audio_period",
]
