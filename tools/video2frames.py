from pathlib import Path
import subprocess

video_dir = Path("/home/liuyang/datasets/damao")

videos = [f for f in video_dir.iterdir() if str(f).lower().endswith('mp4')]


for f in videos:
    result_dir = video_dir / f.stem
    result_dir.mkdir()
    subprocess.run(["ffmpeg", "-i", str(f), "-q:v", "2", f"{str(result_dir / 'frame_%04d.jpg')}"])
