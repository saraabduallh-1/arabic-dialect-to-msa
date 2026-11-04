# Colab overnight runner (baseline only)
# Paste this entire cell after mounting Drive and cloning the repository.
# It backs up checkpoints and logs every 60 seconds, and resumes at the
# experiment level (not mid-epoch). It cannot prevent Colab disconnections.
from pathlib import Path
import os, sys, time, json, shutil, subprocess, threading, datetime

ROOT = Path('/content/arabic-dialect-to-msa')
BACKUP = Path('/content/drive/MyDrive/arabic-dialect-to-msa-experiments/overnight')
assert ROOT.exists(), 'Clone the repository first'
assert Path('/content/drive/MyDrive').exists(), 'Mount Google Drive first'
BACKUP.mkdir(parents=True, exist_ok=True)
RUNS = [
  {'name':'arat5_3epochs', 'model':'UBC-NLP/AraT5v2-base-1024', 'epochs':3, 'bs':4, 'tag':'+overnight-e3'},
  {'name':'arat5_5epochs', 'model':'UBC-NLP/AraT5v2-base-1024', 'epochs':5, 'bs':4, 'tag':'+overnight-e5'},
]
# Each run uses the existing training data; this does not generate dictionary data.
def sync():
    for rel in ['results/scores.jsonl','results/scores_dev.jsonl']:
        source=ROOT/rel
        if source.exists():
            target=BACKUP/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    for name in ['AraT5v2-base-1024+overnight-e3','AraT5v2-base-1024+overnight-e5']:
        source=ROOT/'results/checkpoints'/name
        if source.exists():
            shutil.copytree(source,BACKUP/'checkpoints'/name,dirs_exist_ok=True)
    for p in (ROOT/'results/preds').glob('*overnight*') if (ROOT/'results/preds').exists() else []:
        target=BACKUP/'preds'/p.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)

for run in RUNS:
    done=BACKUP/(run['name']+'.done')
    if done.exists():
        print('SKIP completed',run['name'],flush=True)
        continue
    print('START',run['name'],flush=True)
    stop=threading.Event()
    def periodic():
        while not stop.wait(60):
            try: sync()
            except Exception as e: print('Backup warning:',e,flush=True)
    thread=threading.Thread(target=periodic,daemon=True)
    thread.start()
    cmd=[sys.executable,'src/finetune.py','--model',run['model'],'--epochs',str(run['epochs']),
         '--bs',str(run['bs']),'--optim','adafactor','--tag',run['tag']]
    logfile=BACKUP/(run['name']+'.log')
    with logfile.open('a',encoding='utf-8') as log:
        log.write('\nSTART '+datetime.datetime.now().isoformat()+'\n');log.flush()
        result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    stop.set();thread.join(timeout=5)
    sync()
    if result.returncode:
        print('FAILED',run['name'],'code',result.returncode,'log',logfile,flush=True)
        raise RuntimeError('Training stopped; inspect Drive log before rerunning.')
    done.write_text(datetime.datetime.now().isoformat(),encoding='utf-8')
    print('DONE',run['name'],flush=True)
print('All scheduled baseline runs finished. This cell will not run forever.')
