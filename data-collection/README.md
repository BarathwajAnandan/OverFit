# Why is this here?
This folder contains attempts at using standalone models at solving open-source 
issues. The point is to demonstrate how individual model fails catastrophically 
at solving issues without a shared knowledge base.

# KG representation: 
```
Model → produced → Conversation → about → Issue
                           └─ has_turn → Turn(0) →next→ Turn(1) → ...
```

# Used model 
qwen2-0.5b. 
Other models (--models): starcoder2-3b deepseek-coder-1.3b replit-code-3b.


# To generate KG run:
This pulls 10 issues from https://github.com/pytorch/pytorch/issues/
use `qwen2-0.5b`, (try to) generate 9 Q&A pairs per issue per model. 
```
python pytorch_kb_pipeline.py \
  --repo pytorch/pytorch \
  --limit 10 \
  --models qwen2-0.5b \
  --turns 1 \
  --qna-per-issue 3 \
  --qna-turns 3 \
  --sleep 2.0 \
  --gh-timeout 60 \
  --gh-retries 6 \
  --gh-backoff 0.8 \
  --out_jsonl runs_small.jsonl \
  --out_kg kg_small.json
```