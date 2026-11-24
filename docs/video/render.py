import subprocess, sys, pathlib
from playwright.sync_api import sync_playwright
import imageio_ffmpeg, json, wave
import numpy as np
from sfx import mix
here = pathlib.Path(__file__).parent
page = sys.argv[2] if len(sys.argv) > 2 else "video.html"
base = sys.argv[3] if len(sys.argv) > 3 else "khassis_dialect_msa"
url = (here / page).as_uri()
mode = sys.argv[1]  # preview t1,t2,... | video
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(url + "?t=0"); pg.wait_for_load_state("networkidle"); pg.evaluate("document.fonts.ready")
    if mode == "audio":  # re-mix the sound only
        mode = "video_audio_only"
    if mode == "video":
        fps = 30; n = int(pg.evaluate("DUR") * fps)
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", str(here / f"{base}.mp4")],
                              stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for i in range(n):
            pg.evaluate(f"render({i / fps})")
            ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
        ff.stdin.close(); ff.wait()
        pg.goto(url + "?t=0"); pg.wait_for_load_state("networkidle")
        ev = pg.evaluate("events()")
        mix(ev, n / fps, here / "sfx.wav")
        silent = here / f"{base}.mp4"; out = here / f"{base}_sfx.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(silent), "-i", str(here / "sfx.wav"), "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)], stderr=subprocess.DEVNULL, check=True)
        print("done", n, "events", len(ev))
    else:
        for t in mode.split(","):
            pg.evaluate(f"render({t})"); pg.screenshot(path=str(here / f"f_{t}.png"))
    b.close()
