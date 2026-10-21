"""Unattended runner for the whole benchmark.

1. Waits until the KASS LAYER job (arabic_auto.py) has finished, then 30 more minutes.
2. Installs dependencies, pulls the Ollama models, runs every LLM benchmark,
   fine-tunes the seq2seq models, and writes the report.
3. Commits and pushes to GitHub after every step (push failures are retried later).

Every step is resumable: finished steps are recorded in results/autorun_state.json,
so re-running this script (e.g. after a reboot) continues where it stopped.

usage: .venv/Scripts/python src/autorun.py
"""
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
OLLAMA = str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "ollama.exe")
RESULTS = ROOT / "results"
LOG = RESULTS / "autorun.log"
STATE = RESULTS / "autorun_state.json"
LOCK = RESULTS / "autorun.lock"
TASK_NAME = "ArabicDialectAutorun"

WAIT_FOR = "arabic_auto.py"
# written by the KASS LAYER autopilot once all Gemma runs and its decision test are finished
GPU_FREE_FLAG = ROOT.parent / "kass-layer" / "logs" / "gpu-free.flag"
QUIET_MINUTES = 30

LLMS = ["gemma3:12b", "qwen2.5:7b", "aya-expanse:8b", "command-r7b-arabic:7b",
        "llama3.1:8b", "qwen3:8b", "gemma3:4b", "qwen3.5:9b", "qwen3:14b"]
# larger than the 10 GB GPU (runs partly on CPU, slow) - run last so it can't delay the rest
BIG_LLMS = ["qwen3.8:27b"]
SEQ2SEQ = [("UBC-NLP/AraT5v2-base-1024", "3e-4"), ("moussaKam/AraBART", "5e-5"), ("google/mt5-base", "5e-4")]
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(cmd, timeout=None, check=True):
    log("$ " + " ".join(cmd))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "GIT_TERMINAL_PROMPT": "0"}
    with LOG.open("a", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env,
                           timeout=timeout, creationflags=NO_WINDOW)
    if check and p.returncode:
        raise RuntimeError(f"exit code {p.returncode}: {' '.join(cmd)}")
    return p.returncode


def state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def mark(step):
    s = state()
    s[step] = datetime.now().isoformat(timespec="seconds")
    STATE.write_text(json.dumps(s, indent=1), encoding="utf-8")


def kass_running():
    import psutil
    for p in psutil.process_iter(["cmdline"]):
        try:
            if any(WAIT_FOR in a for a in p.info["cmdline"] or []):
                return True
        except (psutil.Error, TypeError):
            pass
    return False


def wait_for_kass():
    if "wait" in state():
        return
    log(f"waiting for {GPU_FREE_FLAG} and for {WAIT_FOR} to finish")
    quiet_since = None
    while True:
        if not GPU_FREE_FLAG.exists() or kass_running():
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
            log(f"GPU-free flag found and {WAIT_FOR} not running; waiting {QUIET_MINUTES} min to be sure")
        elif time.time() - quiet_since >= QUIET_MINUTES * 60:
            break
        time.sleep(60)
    mark("wait")


def ollama_up():
    import requests
    for _ in range(30):
        try:
            requests.get("http://localhost:11434/api/tags", timeout=5)
            return
        except requests.RequestException:
            subprocess.Popen([OLLAMA, "serve"], creationflags=NO_WINDOW,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(20)
    raise RuntimeError("Ollama server did not start")


def git_sync(message):
    sh(["git", "add", "-A", "src", "results", "data", "README.md", ".gitignore"], check=False)
    sh(["git", "commit", "-q", "-m", message + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
       check=False)
    if sh(["git", "remote", "get-url", "origin"], check=False):
        log("no 'origin' remote configured - committed locally only")
    elif sh(["git", "push", "-q", "-u", "origin", "main"], timeout=300, check=False):
        log("push failed (repo missing or offline?) - will retry after the next step")


def step(name, fn, tries=3):
    if name in state():
        return
    for t in range(1, tries + 1):
        try:
            log(f"=== {name} (try {t})")
            fn()
            mark(name)
            git_sync(f"Benchmark: {name}")
            return
        except Exception:
            log(traceback.format_exc())
            time.sleep(120 * t)
    log(f"!!! giving up on {name}, continuing with the next step")
    git_sync(f"Benchmark: {name} failed (see results/autorun.log)")


def unload_ollama():
    import requests
    for m in requests.get("http://localhost:11434/api/ps", timeout=10).json().get("models", []):
        requests.post("http://localhost:11434/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)


def install():
    pip = [PY, "-m", "pip", "install", "-q", "--timeout", "120", "--retries", "10"]
    sh(pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cu124"])
    sh(pip + ["transformers<5", "sentencepiece", "accelerate", "protobuf", "tiktoken"])


def main():
    RESULTS.mkdir(exist_ok=True)
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 600:
        print("another autorun is active"), sys.exit()
    LOCK.touch()
    import threading

    def heartbeat():
        while True:
            LOCK.touch()
            time.sleep(120)
    threading.Thread(target=heartbeat, daemon=True).start()

    log("autorun started")

    wait_for_kass()
    step("install", install)
    ollama_up()
    for m in LLMS + BIG_LLMS:
        step(f"pull {m}", lambda m=m: sh([OLLAMA, "pull", m], timeout=3 * 3600))

    def llm_steps(models):
        for m in models:
            for mode in ["zero", "rag", "rag_lex"]:
                step(f"llm {m} {mode}", lambda m=m, mode=mode: (
                    ollama_up(), sh([PY, "src/run_llm.py", "--models", m, "--modes", mode], timeout=24 * 3600)))

    llm_steps(LLMS)
    for model, lr in SEQ2SEQ:
        step(f"finetune {model}", lambda model=model, lr=lr: (
            unload_ollama(), sh([PY, "src/finetune.py", "--model", model, "--lr", lr], timeout=12 * 3600)))
    step("report", lambda: sh([PY, "src/report.py"]), tries=1)
    llm_steps(BIG_LLMS)
    step("final report", lambda: sh([PY, "src/report.py"]), tries=1)

    log("ALL DONE")
    git_sync("Benchmark: all done")
    sh(["schtasks", "/delete", "/tn", TASK_NAME, "/f"], check=False)
    LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
