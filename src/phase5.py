"""Phase 5: ALLaM-7B QLoRA on the word task with the enlarged verified data (lahajatna + Darsouni batches).

usage: .venv\Scripts\pythonw src\phase5.py
"""
from autorun import PY, git_sync, log, ollama_up, sh, step, unload_ollama


def main():
    log("phase 5 started")
    step("p5 qlora word ALLaM-AI/ALLaM-7B-Instruct-preview +gen4", lambda: (unload_ollama(), sh(
        [PY, "src/qlora.py", "--model", "ALLaM-AI/ALLaM-7B-Instruct-preview", "--task", "word", "--data", "sentences_v2",
         "--extra", "generated/verified", "--tag", "+gen4"], timeout=48 * 3600)))
    step("p5 word report", lambda: (ollama_up(), sh([PY, "src/word_report.py", "--judge", "qwen3.8:27b"],
                                                    timeout=48 * 3600)), tries=2)
    log("PHASE 5 DONE")
    git_sync("Phase 5: all done")


if __name__ == "__main__":
    main()
