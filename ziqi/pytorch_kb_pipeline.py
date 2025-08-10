# pytorch_kb_pipeline.py
# Scrape GitHub issues, classify type, detect fixes, extract solution/RCA,
# run small coding models on HF, and emit a knowledge graph JSON.

import os
import re
import time
import json
import argparse
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from huggingface_hub import InferenceClient
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GH_TOKEN = os.getenv("GITHUB_TOKEN")

# ---------- Models (small, free-tier friendly) ----------
# nickname: (hf_repo_id, family, param_size)
MODELS = {
    "deepseek-coder-1.3b": ("deepseek-ai/deepseek-coder-1.3b-instruct", "DeepSeek-Coder", "1.3B"),
    "deepseek-coder-6.7b": ("deepseek-ai/deepseek-coder-6.7b-instruct", "DeepSeek-Coder", "6.7B"),
    "starcoder2-3b":       ("bigcode/starcoder2-3b",                    "StarCoder2",      "3B"),
    "phi-3-mini":          ("microsoft/Phi-3-mini-128k-instruct",       "Phi-3",           "3.8B"),
    "replit-code-3b":      ("replit/replit-code-v1-3b",                 "Replit-Code",     "3B"),
    "codegen2-1b":         ("Salesforce/codegen2-1B",                   "CodeGen2",        "1B"),
    # Optional Qwen swap: the *base* (non-instruct) variant exposes text-generation more reliably
    # "qwen2.5-coder-1.5b":  ("Qwen/Qwen2.5-Coder-1.5B",                  "Qwen2.5-Coder",   "1.5B"),
}

# ---------- GitHub helpers ----------
GH_API = "https://api.github.com"
GH_HEADERS = {"Accept": "application/vnd.github+json"}
if GH_TOKEN:
    GH_HEADERS["Authorization"] = f"Bearer {GH_TOKEN}"

def gh_get(url: str, params: Optional[dict] = None):
    r = requests.get(url, headers=GH_HEADERS, params=params or {}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API error {r.status_code} for {url}: {r.text[:500]}")
    return r.json()

def fetch_issues(repo: str, limit: int, state: str = "all", labels: Optional[str] = None) -> List[dict]:
    """Fetch latest issues (skip PRs) with pagination."""
    url = f"{GH_API}/repos/{repo}/issues"
    params = {
        "state": state,
        "per_page": min(limit, 100),
        "page": 1,
        "labels": labels or "",
        "sort": "created",
        "direction": "desc",
    }
    out: List[dict] = []
    while len(out) < limit:
        batch = gh_get(url, params)
        if not batch:
            break
        for it in batch:
            if "pull_request" in it:  # skip PRs
                continue
            out.append(it)
            if len(out) >= limit:
                break
        params["page"] += 1
        time.sleep(0.25)
    return out

def fetch_issue_comments(repo: str, number: int) -> List[dict]:
    url = f"{GH_API}/repos/{repo}/issues/{number}/comments"
    comments: List[dict] = []
    page = 1
    while True:
        batch = gh_get(url, {"page": page, "per_page": 100})
        if not batch:
            break
        comments.extend(batch)
        page += 1
        time.sleep(0.15)
    return comments

# ---------- Issue Type Classification ----------
class IssueType(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    QUESTION = "question"
    DOCS = "docs"
    DISCUSSION = "discussion"
    TASK = "task"
    OTHER = "other"

FEATURE_LABELS = {"feature", "enhancement", "rfc", "proposal", "feature request"}
BUG_LABELS = {"bug", "regression", "crash", "incorrect", "failure"}
QUESTION_LABELS = {"question", "howto", "support", "usage"}
DOC_LABELS = {"documentation", "docs"}
DISCUSS_LABELS = {"discussion", "design discussion"}
TASK_LABELS = {"build", "infra", "ci", "chore"}

def classify_issue(title: str, labels: List[str], body: str) -> IssueType:
    L = {l.lower() for l in labels}
    t = (title or "").lower()
    b = (body or "").lower()

    if L & FEATURE_LABELS or any(k in t for k in ["feature", "rfc", "proposal", "add support", "support for"]):
        return IssueType.FEATURE
    if L & BUG_LABELS or any(k in t for k in ["bug", "regression", "crash", "segfault", "wrong result"]):
        return IssueType.BUG
    if L & QUESTION_LABELS or any(k in t for k in ["how to", "how do i", "question:", "usage"]):
        return IssueType.QUESTION
    if L & DOC_LABELS or "documentation" in t or "docs" in t:
        return IssueType.DOCS
    if L & DISCUSS_LABELS or "rfc" in t or "design" in t:
        return IssueType.DISCUSSION
    if L & TASK_LABELS:
        return IssueType.TASK
    if "proposal" in b or "rfc" in b:
        return IssueType.FEATURE
    return IssueType.OTHER

# ---------- “Fixed” & solution heuristics ----------
FIX_PHRASES = [
    "fixed by", "closed by", "resolved in", "landed in", "merged in",
    "this was fixed", "should be fixed", "will be fixed", "implemented",
    "added support", "addressed in", "available in"
]
SOLUTION_HOOKS = [
    "workaround", "solution", "fix", "patch", "proposed change",
    "apply this", "use this", "add this", "upgrade to", "revert", "cherry-pick"
]

def is_fixed(issue: dict, comments: List[dict], itype: IssueType) -> bool:
    # Closed counts as addressed; type refines textual signals
    if issue.get("state") == "closed":
        return True
    text = (issue.get("title") or "") + "\n" + (issue.get("body") or "")
    text += "\n".join((c.get("body") or "") for c in comments)
    ltext = text.lower()

    if itype == IssueType.BUG:
        return any(p in ltext for p in FIX_PHRASES)
    if itype == IssueType.FEATURE:
        return any(p in ltext for p in ["implemented", "added support", "landed in", "merged in", "available in"])
    # For other types, rely on closed-state primarily
    return False

def extract_solution_and_root_cause(comments: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Select a likely maintainer/collaborator solution comment, then pull a plausible RCA sentence.
    """
    maintainer_roles = {"MEMBER", "OWNER", "COLLABORATOR"}
    candidates: List[Tuple[int, str]] = []
    for c in comments:
        role = (c.get("author_association") or "").upper()
        body = (c.get("body") or "").strip()
        if role in maintainer_roles and len(body) > 30:
            score = 0
            if "```" in body or "~~~" in body:
                score += 2
            if any(h in body.lower() for h in SOLUTION_HOOKS):
                score += 2
            if len(body) > 200:
                score += 1
            candidates.append((score, body))

    candidates.sort(reverse=True, key=lambda x: x[0])
    solution = candidates[0][1] if candidates else None

    rca = None
    if solution:
        # crude RCA extraction
        for ln in re.split(r"[.\n]", solution):
            l = ln.strip().lower()
            if any(k in l for k in ["because", "due to", "caused by", "root cause", "reason is", "happens when"]):
                rca = ln.strip()
                break
    return solution, rca

# ---------- LLM calls ----------
def build_prompt(issue: dict, itype: IssueType) -> str:
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    url = issue.get("html_url", "")
    body_trunc = body[:3000]

    if itype == IssueType.BUG:
        task = """Provide:
1) Likely root cause (usage vs library bug vs env/build) with justification.
2) Minimal repro steps or code.
3) Proposed fix/workaround (code/config).
4) Relevant PyTorch docs/files (best effort)."""
    elif itype == IssueType.FEATURE:
        task = """Treat as a feature request/RFC. Provide:
1) Scope & rationale.
2) Minimal API or design sketch.
3) Back-compat considerations.
4) Implementation plan (files/modules) + test strategy.
5) Short-term workaround if not implemented."""
    elif itype == IssueType.QUESTION:
        task = """Treat as a user question. Provide:
1) Likely intent/misunderstanding.
2) Correct usage with a small example.
3) Links to relevant docs."""
    else:
        task = "Provide concise next steps, likely category, and relevant references."

    return f"""You are triaging a PyTorch GitHub item.

Issue: {url}
Title: {title}

Body (truncated):
{body_trunc}

TASK:
{task}
Keep it concise and actionable.
"""

# --- compat shim for get_model_info ---
try:
    # Newer huggingface_hub
    from huggingface_hub import get_model_info

    @lru_cache(maxsize=128)
    def _pipeline_tag(model_id: str) -> str:
        try:
            info = get_model_info(model_id)
            return (getattr(info, "pipeline_tag", "") or "")  # empty if missing
        except Exception:
            return ""
except ImportError:
    # Older huggingface_hub
    from huggingface_hub import HfApi
    _HF_API = HfApi()

    @lru_cache(maxsize=128)
    def _pipeline_tag(model_id: str) -> str:
        try:
            info = _HF_API.model_info(model_id)  # returns ModelInfo
            # Some older versions don’t set .pipeline_tag; try cardData fallback
            tag = getattr(info, "pipeline_tag", None)
            if not tag:
                card_data = getattr(info, "cardData", None) or {}
                tag = card_data.get("pipeline_tag", "")
            return tag or ""
        except Exception:
            return ""

def _as_messages(system_prompt: str, user_prompt: str):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs

def call_hf(repo_id: str, prompt: str,
            temperature: float = 0.2,
            max_new_tokens: int = 500,
            system_prompt: str = "You are a concise, accurate PyTorch triage assistant.") -> str:
    """
    Robust call that:
      1) checks the model's pipeline tag,
      2) uses text_generation if available,
      3) otherwise falls back to chat_completion,
      4) retries once with safer settings on transient errors.
    """
    client = InferenceClient(model=repo_id, token=HF_TOKEN, timeout=120)
    tag = _pipeline_tag(repo_id).lower()

    def _try_text():
        return client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05,
            return_full_text=False,
            # wait_for_model=True   # uncomment if you frequently hit cold starts
        )

    def _try_chat():
        resp = client.chat_completion(
            messages=_as_messages(system_prompt, prompt),
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        # HF returns an OpenAI-like shape
        return resp.choices[0].message["content"]

    # Preferred path based on pipeline tag
    try_order = []
    if tag == "text-generation" or tag == "text2text-generation":
        try_order = ["text", "chat"]
    elif tag == "conversational" or tag == "chat":
        try_order = ["chat", "text"]
    else:
        # unknown: try text first, then chat
        try_order = ["text", "chat"]

    for attempt, mode in enumerate(try_order + try_order[:1]):  # allow one extra retry with safer params
        try:
            if mode == "text":
                out = _try_text()
                return out if isinstance(out, str) else getattr(out, "generated_text", str(out))
            else:
                return _try_chat()
        except StopIteration as e:
            # Often a provider-side empty stream — short backoff and retry next mode
            time.sleep(0.8)
            continue
        except ValueError as e:
            # Typical “task not supported” message — switch mode
            if "not supported for task text-generation" in str(e).lower() and "conversational" in str(e).lower():
                # force chat next
                if "chat" in try_order:
                    continue
            # Any other ValueError: try the other path
            continue
        except Exception as e:
            # One final conservative retry with no sampling and fewer tokens
            if attempt == 0:
                try:
                    if mode == "text":
                        out = client.text_generation(
                            prompt,
                            max_new_tokens=min(256, max_new_tokens),
                            temperature=0.0,
                            do_sample=False,
                            return_full_text=False,
                        )
                        return out if isinstance(out, str) else getattr(out, "generated_text", str(out))
                    else:
                        resp = client.chat_completion(
                            messages=_as_messages(system_prompt, prompt),
                            max_tokens=min(256, max_new_tokens),
                            temperature=0.0,
                        )
                        return resp.choices[0].message["content"]
                except Exception:
                    pass
            # move on to next mode
            continue

    return "[MODEL ERROR] All attempts failed (text/chat). Check model task support or rate limits."

# ---------- Knowledge Graph helpers ----------
def add_node(nodes: Dict[str, dict], node_id: str, label: str, ntype: str, props: Optional[dict] = None):
    if node_id in nodes:
        return
    nodes[node_id] = {"id": node_id, "label": label, "type": ntype, "props": props or {}}

def add_edge(edges: List[dict], src: str, dst: str, etype: str, props: Optional[dict] = None):
    edges.append({"from": src, "to": dst, "type": etype, "props": props or {}})

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="PyTorch issues -> type-aware triage + small-model answers -> KG JSON")
    ap.add_argument("--repo", default="pytorch/pytorch", help="owner/repo (default: pytorch/pytorch)")
    ap.add_argument("--limit", type=int, default=8, help="number of issues to process")
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                    help=f"subset of models (default: {', '.join(MODELS.keys())})")
    ap.add_argument("--sleep", type=float, default=1.0, help="sleep seconds between model calls")
    ap.add_argument("--out_jsonl", default="runs.jsonl", help="raw per-issue results (JSONL)")
    ap.add_argument("--out_kg", default="knowledge_graph.json", help="knowledge graph output (JSON)")
    args = ap.parse_args()

    issues = fetch_issues(args.repo, args.limit, state="all")
    print(f"Fetched {len(issues)} issues from {args.repo}")

    nodes: Dict[str, dict] = {}
    edges: List[dict] = []

    # Seed model family nodes
    for nick in args.models:
        repo_id, family, size = MODELS[nick]
        model_node_id = f"model::{family}::{size}"
        add_node(nodes, model_node_id, f"{family} {size}", "Model",
                 {"nickname": nick, "hf_repo": repo_id})

    jl = open(args.out_jsonl, "a", encoding="utf-8")

    for ishu in issues:
        number = ishu["number"]
        issue_url = ishu["html_url"]
        title = ishu.get("title", "") or ""
        body = ishu.get("body", "") or ""
        labels = [l["name"] for l in ishu.get("labels", [])]

        print(f"\n→ Issue #{number}: {title}")
        comments = fetch_issue_comments(args.repo, number)

        itype = classify_issue(title, labels, body)
        fixed = is_fixed(ishu, comments, itype)
        solution_text, rca_text = extract_solution_and_root_cause(comments)

        # Issue node
        issue_node_id = f"issue::{args.repo}#{number}"
        add_node(nodes, issue_node_id, f"{args.repo}#{number}", "Issue", {
            "title": title,
            "url": issue_url,
            "state": ishu.get("state", ""),
            "type": itype.value,
            "created_at": ishu.get("created_at", ""),
            "updated_at": ishu.get("updated_at", ""),
            "labels": labels
        })

        # Solution / RCA nodes
        solution_node_id = None
        if solution_text:
            solution_node_id = f"solution::{args.repo}#{number}"
            add_node(nodes, solution_node_id, f"Solution for {args.repo}#{number}", "Solution",
                     {"text": solution_text})
            edge_type = "implements" if itype == IssueType.FEATURE else "solves"
            add_edge(edges, solution_node_id, issue_node_id, edge_type)

        rca_node_id = None
        if rca_text:
            rca_node_id = f"rca::{args.repo}#{number}"
            add_node(nodes, rca_node_id, f"RCA for {args.repo}#{number}", "RootCause", {"text": rca_text})
            add_edge(edges, rca_node_id, issue_node_id, "causes")

        # If marked addressed but no extracted solution, add placeholder for later curation
        if fixed and not solution_text:
            placeholder = "Solution likely in linked PR/commit/comments; needs manual curation."
            solution_node_id = f"solution::{args.repo}#{number}"
            if solution_node_id not in nodes:
                add_node(nodes, solution_node_id, f"Solution for {args.repo}#{number}", "Solution",
                         {"text": placeholder})
                edge_type = "implements" if itype == IssueType.FEATURE else "solves"
                add_edge(edges, solution_node_id, issue_node_id, edge_type)

        # Build type-aware prompt and query models
        prompt = build_prompt(ishu, itype)
        model_answers: Dict[str, str] = {}
        for nick in args.models:
            repo_id, family, size = MODELS[nick]
            ans = call_hf(repo_id, prompt)
            model_answers[nick] = ans

            # KG: model -> answer -> issue
            model_node_id = f"model::{family}::{size}"
            ans_node_id = f"answer::{nick}::{args.repo}#{number}"
            add_node(nodes, ans_node_id, f"Answer by {nick} on #{number}", "Answer", {"text": ans})
            add_edge(edges, model_node_id, ans_node_id, "produced")
            add_edge(edges, ans_node_id, issue_node_id, "about")
            add_edge(edges, model_node_id, issue_node_id, "answered")

            time.sleep(args.sleep)

        # Raw record per issue
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "repo": args.repo,
            "issue_number": number,
            "issue_url": issue_url,
            "title": title,
            "labels": labels,
            "type": itype.value,
            "state": ishu.get("state", ""),
            "fixed": fixed,
            "solution_text": solution_text,
            "rca_text": rca_text,
            "models": [
                {
                    "nickname": nick,
                    "family": MODELS[nick][1],
                    "param_size": MODELS[nick][2],
                    "hf_repo": MODELS[nick][0],
                    "answer": model_answers[nick]
                } for nick in args.models
            ]
        }
        jl.write(json.dumps(record, ensure_ascii=False) + "\n")
        jl.flush()

    jl.close()

    # Emit Knowledge Graph JSON
    kg = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": {
            "node_types": ["Model", "Issue", "Solution", "RootCause", "Answer"],
            "edge_types": ["answered", "produced", "about", "solves", "implements", "causes"]
        },
        "nodes": list(nodes.values()),
        "edges": edges
    }
    with open(args.out_kg, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    print(f"\nDone.\n- Raw per-issue runs: {args.out_jsonl}\n- Knowledge graph JSON: {args.out_kg}")

if __name__ == "__main__":
    main()