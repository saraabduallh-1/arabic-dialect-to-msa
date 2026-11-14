"""Phase 4 (unattended): judge check against the user's 150 labels, ALLaM, and the verified generated data.

Part A (no new data needed): judge agreement with results/human_labels.jsonl, ALLaM-7B (Saudi, SDAIA)
on the word-level task.
Part B (runs once data/generated/verified.csv exists): word-level prompting with the verified data,
then QLoRA on the word-level task.
Steps are resumable and each one commits and pushes (helpers from autorun.py).

usage: .venv\\Scripts\\python src\\phase4.py
"""
from autorun import PY, git_sync, log, ollama_up, sh, step, unload_ollama
from data import DATA
from phase3 import GEN2, word

VERIFIED = DATA / "generated" / "verified.csv"


def main():
    log("phase 4 started")
    step("p4 judge agreement human", lambda: (ollama_up(), sh(
        [PY, "src/judge.py", "--judge", "qwen3.8:27b", "--agreement-only"], timeout=6 * 3600)))
    step("p4 word allam:7b zero", lambda: word("allam:7b", "zero", []))
    step("p4 word allam:7b rag+gen2", lambda: word("allam:7b", "rag", GEN2))
    if not VERIFIED.exists():
        log("phase 4 part A done; part B waits for data/generated/verified.csv")
        git_sync("Phase 4 part A done")
        return

    gen3 = ["--extra", "generated/verified", "--tag", "+gen3"]
    for m in ("gemma3:27b", "allam:7b"):
        step(f"p4 word {m} rag+gen3", lambda m=m: word(m, "rag", gen3))
    for m in ("ALLaM-AI/ALLaM-7B-Instruct-preview", "Qwen/Qwen2.5-7B-Instruct"):
        step(f"p4 qlora word {m}", lambda m=m: (unload_ollama(), sh(
            [PY, "src/qlora.py", "--model", m, "--task", "word", "--data", "sentences_v2",
             "--extra", "generated/verified", "--tag", "+gen3"], timeout=48 * 3600)))
    step("p4 word report", lambda: (ollama_up(), sh([PY, "src/word_report.py", "--judge", "qwen3.8:27b"],
                                                    timeout=48 * 3600)), tries=2)
    log("PHASE 4 DONE")
    git_sync("Phase 4: all done")


if __name__ == "__main__":
    main()
