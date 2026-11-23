import subprocess, sys, pathlib
from playwright.sync_api import sync_playwright
import imageio_ffmpeg
here = pathlib.Path(__file__).parent
url = (here / "video.html").as_uri()
mode = sys.argv[1]  # preview t1,t2,... | video
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(url + "?t=0"); pg.wait_for_load_state("networkidle"); pg.evaluate("document.fonts.ready")
    if mode == "video":
        fps = 30; n = int(pg.evaluate("DUR") * fps)
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", str(here / "khassis_dialect_msa.mp4")],
                              stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for i in range(n):
            pg.evaluate(f"render({i / fps})")
            ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
        ff.stdin.close(); ff.wait(); print("done", n)
    else:
        for t in mode.split(","):
            pg.evaluate(f"render({t})"); pg.screenshot(path=str(here / f"f_{t}.png"))
    b.close()
