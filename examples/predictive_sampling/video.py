import os
import shutil
import subprocess
from collections import namedtuple
from datetime import datetime


VideoRecorder = namedtuple("VideoRecorder", ["process", "video_path"])


def start_recording(output_dir, width, height, fps):
    if shutil.which("ffmpeg") is None:
        print("Warning: ffmpeg not found. Video recording disabled.")
        return None
    os.makedirs(output_dir, exist_ok=True)
    video_path = build_video_path(output_dir)
    cmd = build_ffmpeg_cmd(width, height, fps, video_path)
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    print(f"Recording video to {video_path}")
    return VideoRecorder(process, video_path)


def build_video_path(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"simulation_{timestamp}.mp4")


def build_ffmpeg_cmd(width, height, fps, video_path):
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo"]
    cmd += ["-s", f"{width}x{height}", "-pix_fmt", "rgb24"]
    cmd += ["-r", str(fps), "-i", "-", "-an"]
    cmd += ["-vcodec", "h264", "-crf", "1", "-preset", "slow"]
    cmd += ["-movflags", "+faststart", "-pix_fmt", "yuv420p"]
    cmd += ["-profile:v", "high", "-tune", "film"]
    cmd += ["-loglevel", "error", video_path]
    return cmd


def add_frame(recorder, frame):
    try:
        recorder.process.stdin.write(frame)
    except (BrokenPipeError, IOError):
        pass


def stop_recording(recorder):
    try:
        recorder.process.stdin.close()
        recorder.process.wait()
        print(f"Video saved to {recorder.video_path}")
    except (subprocess.TimeoutExpired, BrokenPipeError, IOError) as error:
        print(f"Warning: Error finalizing video: {error}")
        recorder.process.terminate()
