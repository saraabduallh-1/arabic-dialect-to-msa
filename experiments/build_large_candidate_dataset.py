#!/usr/bin/env python3
"""Generate auditable synthetic candidates grounded in the project's TRAIN split.
No dictionary text is copied. Never use dev/test to construct candidates.
Run: python experiments/build_large_candidate_dataset.py
"""
import csv, json, random, re, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'augmentation'
OUT.mkdir(parents=True,exist_ok=True)
random.seed(20261105)
import sys
sys.path.insert(0,str(ROOT/'src'))
from data import load, normalize
train=load('train')
dev=load('dev')
test=load('test')
# Sentence frames with neutral content. All generated examples are synthetic and unreviewed.
# One marked span is inserted at a time, preserving source-target alignment.
FRAMES=[
 ('قال لي: «{x}».','قال لي: «{y}».'),
 ('كتبت في الرسالة: {x}.','كتبت في الرسالة: {y}.'),
 ('سمعت أحدهم يقول: {x}.','سمعت أحدهم يقول: {y}.'),
 ('في المحادثة ظهرت عبارة «{x}».','في المحادثة ظهرت عبارة «{y}».'),
 ('كانت الكلمة المستخدمة هي «{x}».','كانت الكلمة المستخدمة هي «{y}».'),
 ('ردّ علي بكلمة «{x}».','ردّ علي بكلمة «{y}».'),
]
# Contexts that explicitly mention words/phrases, rather than falsely pretending
# every dialect term fits every grammatical slot.
seen={}
for r in train:
 for dialect,msa in r['pairs']:
  dialect=dialect.strip(); msa=msa.strip()
  if not dialect or not msa or dialect==msa:continue
  key=(normalize(dialect),normalize(msa))
  if key not in seen:seen[key]=(dialect,msa,r['id'])
all_refs={normalize(r['src']) for r in dev+test}|{normalize(r['tgt']) for r in dev+test}
train_text={normalize(r['src']) for r in train}|{normalize(r['tgt']) for r in train}
candidates=[]
used=set()
for (norm_d,norm_m),(d,m,source_id) in sorted(seen.items()):
 for frame_idx,(f1,f2) in enumerate(FRAMES):
  s=f1.format(x=d); t=f2.format(y=m)
  if normalize(s)==normalize(t):continue
  key=(normalize(s),normalize(t))
  if key in used or normalize(s) in all_refs or normalize(t) in all_refs:continue
  if normalize(s) in train_text or normalize(t) in train_text:continue
  used.add(key)
  candidates.append({
   'id':'syn-'+hashlib.sha256((str(source_id)+'|'+str(frame_idx)+'|'+d+'|'+m).encode()).hexdigest()[:16],
   'dialect':s,'msa':t,'dialect_span':d,'msa_span':m,
   'origin_train_id':source_id,'frame_id':frame_idx,
   'provenance':'synthetic_from_project_train_annotation',
   'review_status':'needs_human_review',
   'task_fit':'metalinguistic_context_only',
  })
# This generator deliberately does NOT call these ordinary conversational
# sentences: it only produces quotation/mention contexts. Do not silently
# merge them into training: this distribution is artificial and repetitive.
columns=['id','dialect','msa','dialect_span','msa_span','origin_train_id','frame_id','provenance','review_status','task_fit']
with (OUT/'large_candidates_train_grounded.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(candidates)
report={
 'train_sentences':len(train),'dev_sentences':len(dev),'test_sentences':len(test),
 'unique_train_span_mappings':len(seen),'generated_candidates':len(candidates),
 'source':'project training annotations only; NOT external dictionary-sourced',
 'licensing':'project existing training split; verify upstream provenance',
 'evaluation_contamination':'exact normalized dev/test sentence collisions filtered; semantic overlap not ruled out',
 'training_ready':False,
 'limitations':['templated metalinguistic sentences, not natural dialogue',
                'six frames per lexicon mapping create highly correlated samples',
                'human validation and independent lexical verification required',
                'do not train on these without balancing and review']
}
(OUT/'large_candidates_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
