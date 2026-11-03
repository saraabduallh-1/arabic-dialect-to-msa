"""Phase 3 (unattended): experiments on the corrected data (data/sentences_v2).

1. Word-level task (run_word.py) for the strongest models: zero, rag+gen2, rag_lexf+gen2
   (+gen2 = all 40 generated batches, data/generated/all2.csv, as examples and dictionary).
2. The best sentence-level setup again, trained/prompted on the corrected data, for comparison.
3. Word-level report with the judge (word_report.py).
Steps are resumable and each one commits and pushes (helpers from autorun.py).

usage: .venv\\Scripts\\python src\\phase3.py
"""
from autorun import PY, git_sync, log, ollama_up, sh, step

MODELS = ["gemma3:27b", "qwen3.8:27b", "gemma3:12b", "command-r7b-arabic:7b"]
GEN2 = ["--extra", "generated/all2", "--tag", "+gen2"]


def word(model, mode, extra, k=8):
    ollama_up()
    sh([PY, "src/run_word.py", "--models", model, "--modes", mode, "--data", "sentences_v2", "--k", str(k)] + extra,
       timeout=24 * 3600)


def main():
    log("phase 3 started (corrected data, word-level task)")
    for m in MODELS:
        step(f"p3 word {m} zero", lambda m=m: word(m, "zero", []))
        step(f"p3 word {m} rag+gen2", lambda m=m: word(m, "rag", GEN2))
        step(f"p3 word {m} rag_lexf+gen2", lambda m=m: word(m, "rag_lexf", GEN2))
    step("p3 word gemma3:27b rag_lexf+gen2 k16",
         lambda: word("gemma3:27b", "rag_lexf", ["--extra", "generated/all2", "--tag", "+gen2_k16"], k=16))
    step("p3 sentence gemma3:27b rag_lexret v2", lambda: (ollama_up(), sh(
        [PY, "src/run_llm.py", "--models", "gemma3:27b", "--modes", "rag_lexret", "--data", "sentences_v2", "--k", "16",
         "--extra", "generated/all2", "--tag", "+gen2_v2"], timeout=24 * 3600)))
    step("p3 word report", lambda: (ollama_up(), sh([PY, "src/word_report.py", "--judge", "qwen3.8:27b"],
                                                     timeout=48 * 3600)), tries=2)
    log("PHASE 3 DONE")
    git_sync("Phase 3: all done")


if __name__ == "__main__":
    main()
