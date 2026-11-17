"""Synchronize four interview cameras with different starts into a 2x2 grid."""

from moviepy import AudioFileClip, MultiCameraClip, VideoFileClip


# All four feeds are derived from the same example media here. In a real
# project each one would be a different VideoFileClip with its own resolution
# and starting point. Every entry is (clip, offset_in_seconds, (row, col)).
base = VideoFileClip("example.mp4")
cameras = [
    (base, 0.0, (0, 0)),
    (base, 0.25, (0, 1)),  # this camera started 0.25s later
    (base, 0.5, (1, 0)),
    (base, 0.75, (1, 1)),
]

grid = MultiCameraClip(
    cameras=cameras,
    size=(640, 480),  # fixed output resolution, each cell is 320x240
    duration=2,  # fixed output duration
    fit="contain",  # letterbox instead of cropping; use "cover" to crop
    bg_color=(0, 0, 0),  # color of empty cells and letterbox borders
    audio_mode="mix",  # example.mp4 has no audio, so the result has none
)

# A soundtrack can be attached afterwards, for instance a shared voice track.
grid = grid.with_audio(AudioFileClip("example.wav").subclipped(0, grid.duration))

grid.write_videofile("multicamera_grid.mp4", fps=25, logger=None)
grid.close()
base.close()
