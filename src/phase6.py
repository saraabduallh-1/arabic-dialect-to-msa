from autorun import PY, git_sync, log, ollama_up, sh, step, unload_ollama


def main():
    log("phase 6 started")
    step("p6 qlora word ALLaM +gen3 +lex", lambda: (unload_ollama(), sh(
        [PY, "src/qlora.py", "--model", "ALLaM-AI/ALLaM-7B-Instruct-preview", "--task", "word", "--data", "sentences_v2",
         "--extra", "generated/verified", "--lex", "--tag", "+gen4+lex"], timeout=48 * 3600)))
    step("p6 word report", lambda: (ollama_up(), sh([PY, "src/word_report.py", "--judge", "qwen3.8:27b"],
                                                    timeout=48 * 3600)), tries=2)
    log("PHASE 6 DONE")
    git_sync("Phase 6: dictionary hints")


if __name__ == "__main__":
    main()
