#!/usr/bin/env python3
# pytorch_kb_pipeline.py
# Scrape GitHub issues, classify type, detect fixes, extract solution/RCA (incl. closing PR),
# run small coding models on HF as multi-turn chats, record full conversation traces,
# generate prompt-driven Q&A pairs (via a generator LLM) for each conversational model,
# and emit a knowledge graph JSON with Conversation/Turn/Question/Answer nodes (plus JSONL run logs).
# Includes robust GitHub HTTP client (retries/backoff/UA) and HF calls tolerant of empty responses.

import os
import re
import time
import json
import argparse
import random
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import ConnectionError as ReqConnectionError, Timeout as ReqTimeout
from urllib3.exceptions import ProtocolError as Urllib3ProtocolError

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GH_TOKEN = os.getenv("GITHUB_TOKEN")

# ---------- Models (original format: Dict[str, Tuple[repo_id, family, param_size]]) ----------
# Keep only stable text-generation-capable endpoints to minimize StopIteration issues.
MODELS: Dict[str, Tuple[str, str, str]] = {
    "deepseek-coder-1.3b": ("deepseek-ai/deepseek-coder-1.3b-instruct", "DeepSeek-Coder", "1.3B"),
    "starcoder2-3b":  ("bigcode/starcoder2-3b",              "StarCoder2",  "3B"),
    "replit-code-3b": ("replit/replit-code-v1-3b",           "Replit-Code", "3B"),
    "qwen2-0.5b":          ("Qwen/Qwen2-0.5B-Instruct",                 "Qwen2",          "0.5B"),
    # "phi-3-mini":     ("microsoft/Phi-3-mini-128k-instruct", "Phi-3",       "3.8B"),
}

# Generator model (repo_id, family, size) — used as the default for --gen-model
GEN_MODEL = ("bigcode/starcoder2-3b", "StarCoder2", "3B")

# ---------- GitHub helpers (robust session + headers) ----------
GH_API = "https://api.github.com"
GH_HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "pytorch-kb/0.1"}
if GH_TOKEN:
    GH_HEADERS["Authorization"] = f"Bearer {GH_TOKEN}"

# Globals set from CLI
SESSION: requests.Session = None  # built in main()
GH_TIMEOUT: float = 45.0          # default, overridden by --gh-timeout

def _requests_session(retries: int = 5, backoff: float = 0.6) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "pytorch-kb/0.1 (+https://example.local)"})
    return s

def gh_get(url: str, params: Optional[dict] = None, timeout_s: float = None):
    """GET with retries/backoff + one manual jittered retry on connection resets."""
    params = params or {}
    tmo = timeout_s if timeout_s is not None else GH_TIMEOUT
    try:
        r = SESSION.get(url, headers=GH_HEADERS, params=params, timeout=tmo)
        if r.status_code == 429:
            retry_after = float(r.headers.get("Retry-After", "3"))
            time.sleep(retry_after)
            r = SESSION.get(url, headers=GH_HEADERS, params=params, timeout=tmo)
        if 200 <= r.status_code < 300:
            return r.json()
        raise RuntimeError(f"GitHub API {r.status_code} for {url} :: {r.text[:500]}")
    except (ReqConnectionError, ReqTimeout, Urllib3ProtocolError) as e:
        time.sleep(0.5 + random.random())
        r = SESSION.get(url, headers=GH_HEADERS, params=params, timeout=tmo)
        if 200 <= r.status_code < 300:
            return r.json()
        raise RuntimeError(f"GitHub API retry failed {r.status_code} for {url} :: {r.text[:500]} :: {type(e).__name__}: {e}")

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
        batch = gh_get(url, params, timeout_s=GH_TIMEOUT)
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
        batch = gh_get(url, {"page": page, "per_page": 100}, timeout_s=GH_TIMEOUT)
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

# ---------- “Fixed” & solution heuristics (incl. close PR follow) ----------
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
    if issue.get("state") == "closed":
        return True
    text = (issue.get("title") or "") + "\n" + (issue.get("body") or "")
    text += "\n".join((c.get("body") or "") for c in comments)
    ltext = text.lower()
    if itype == IssueType.BUG:
        return any(p in ltext for p in FIX_PHRASES)
    if itype == IssueType.FEATURE:
        return any(p in ltext for p in ["implemented", "added support", "landed in", "merged in", "available in"])
    return False

def extract_solution_and_root_cause(comments: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    maintainer_roles = {"MEMBER", "OWNER", "COLLABORATOR", "CONTRIBUTOR", "TRIAGER"}
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
        for ln in re.split(r"[.\n]", solution):
            l = ln.strip().lower()
            if any(k in l for k in ["because", "due to", "caused by", "root cause", "reason is", "happens when"]):
                rca = ln.strip()
                break
    return solution, rca

CLOSING_KEYWORDS = ["fixes", "closes", "resolves"]

def find_closing_pr(repo: str, issue_number: int) -> Optional[int]:
    q_variants = [
        f'repo:{repo} is:pr is:merged {" OR ".join(k.capitalize() for k in CLOSING_KEYWORDS)} #{issue_number}',
        f'repo:{repo} is:pr is:merged {" OR ".join(CLOSING_KEYWORDS)} #{issue_number}',
        f'repo:{repo} is:pr is:merged "#{issue_number}"',
    ]
    for q in q_variants:
        data = gh_get(f"{GH_API}/search/issues", {"q": q, "per_page": 5}, timeout_s=GH_TIMEOUT)
        items = data.get("items", []) if isinstance(data, dict) else []
        for it in items:
            pr_num = it.get("number")
            if pr_num:
                return pr_num
        time.sleep(0.2)
    return None

def fetch_pr_body_and_commits(repo: str, pr_number: int) -> Tuple[str, List[str]]:
    pr = gh_get(f"{GH_API}/repos/{repo}/pulls/{pr_number}", timeout_s=GH_TIMEOUT)
    body = pr.get("body") or ""
    commits = gh_get(f"{GH_API}/repos/{repo}/pulls/{pr_number}/commits", timeout_s=GH_TIMEOUT)
    msgs = []
    for c in commits[:5]:
        msg = (c.get("commit", {}) or {}).get("message", "")
        if msg:
            msgs.append(msg)
    return body, msgs

def extract_solution_rca_from_texts(texts: List[str]) -> Tuple[Optional[str], Optional[str]]:
    solution = None
    best_score = -1
    for t in texts:
        if not t:
            continue
        score = 0
        tl = t.lower()
        if "```" in t or "~~~" in t:
            score += 2
        if any(w in tl for w in ["workaround", "fix", "patch", "apply", "change", "update", "replace"]):
            score += 2
        if any(k in tl for k in CLOSING_KEYWORDS):
            score += 1
        if len(t) > 200:
            score += 1
        if score > best_score:
            best_score, solution = score, t

    rca = None
    if solution:
        for ln in re.split(r"[.\n]", solution):
            l = ln.strip().lower()
            if any(k in l for k in ["because", "due to", "caused by", "root cause", "reason is", "happens when"]):
                rca = ln.strip()
                break
    return solution, rca

# ---------- Conversation trace structures ----------
@dataclass
class ChatTurn:
    role: str                   # "system" | "user" | "assistant"
    content: str
    ts: str                     # ISO time
    meta: Dict[str, str] = None

@dataclass
class ChatTrace:
    model_id: str
    model_nick: str
    family: str
    size: str
    messages: List[ChatTurn]

    def add(self, role: str, content: str, **meta):
        self.messages.append(ChatTurn(
            role=role,
            content=content,
            ts=datetime.now(timezone.utc).isoformat(),
            meta=meta or {}
        ))

# ---------- Pipeline tag detection ----------
try:
    from huggingface_hub import get_model_info

    @lru_cache(maxsize=128)
    def _pipeline_tag(model_id: str) -> str:
        try:
            info = get_model_info(model_id)
            return (getattr(info, "pipeline_tag", "") or "").lower()
        except Exception:
            return ""
except Exception:
    from huggingface_hub import HfApi
    _api = HfApi()

    @lru_cache(maxsize=128)
    def _pipeline_tag(model_id: str) -> str:
        try:
            info = _api.model_info(model_id)
            tag = getattr(info, "pipeline_tag", None)
            if not tag:
                card = getattr(info, "cardData", None) or {}
                tag = (card.get("pipeline_tag") or "").lower()
            return (tag or "").lower()
        except Exception:
            return ""

# ---------- Hardened single-shot ask (generator) ----------
def _chat_once(model_id: str, messages: List[Dict[str, str]],
               max_new_tokens: int = 400, temperature: float = 0.4) -> str:
    """Ask any HF model (chat or text) robustly. Retries across modes; tolerates empty outputs."""
    tag = _pipeline_tag(model_id)

    def _mk():
        return InferenceClient(model=model_id, token=HF_TOKEN, timeout=120)

    def _flatten(msgs: List[Dict[str, str]]) -> str:
        flat = []
        for m in msgs:
            c = (m.get("content") or "").strip()
            if len(c) > 4500:
                c = c[:4500]
            flat.append(f"{m.get('role','user').upper()}:\n{c}\n")
        flat.append("ASSISTANT:\n")
        return "".join(flat)

    def _try_chat(client, mx, temp, sample):
        resp = client.chat_completion(
            messages=messages,
            max_tokens=mx,
            temperature=temp,
            top_p=0.9 if sample else 1.0,
        )
        if not resp or not getattr(resp, "choices", None):
            raise RuntimeError("empty chat response (no choices)")
        content = resp.choices[0].message.get("content") if hasattr(resp.choices[0], "message") else None
        if not content or not str(content).strip():
            raise RuntimeError("empty chat content")
        return content

    def _try_text(client, mx, temp, sample):
        out = client.text_generation(
            _flatten(messages),
            max_new_tokens=mx,
            temperature=temp,
            top_p=0.9 if sample else 1.0,
            do_sample=sample,
            repetition_penalty=1.05 if sample else 1.0,
            return_full_text=False,
        )
        text = out if isinstance(out, str) else getattr(out, "generated_text", None)
        if not text or not str(text).strip():
            raise RuntimeError("empty text generation")
        return text

    plans = [
        ("chat", temperature, max_new_tokens, True),
        ("text", temperature, max_new_tokens, True),
        ("chat", 0.0, min(256, max_new_tokens), False),
        ("text", 0.0, min(256, max_new_tokens), False),
    ]

    last_err = None
    for i, (mode, temp, mx, sample) in enumerate(plans):
        try:
            client = _mk()
            if (tag in {"conversational", "chat"}) and mode == "chat":
                return _try_chat(client, mx, temp, sample)
            elif mode == "text":
                return _try_text(client, mx, temp, sample)
            else:
                continue
        except (StopIteration, Exception) as e:
            last_err = e
            time.sleep(0.8 + 0.4 * (i + 1))
            continue

    return "[GEN ERROR] empty/aborted response after retries"

# ---------- Chat-or-text caller (records full turns) ----------
def _flatten_messages_for_textgen(messages: List[Dict[str, str]]) -> str:
    chunks = []
    for m in messages:
        c = (m.get("content") or "").strip()
        if len(c) > 4500:
            c = c[:4500]
        chunks.append(f"{m.get('role','user').upper()}:\n{c}\n")
    chunks.append("ASSISTANT:\n")
    return "\n".join(chunks)

def call_hf_chat(repo_id: str,
                 trace: ChatTrace,
                 temperature: float = 0.2,
                 max_new_tokens: int = 500) -> str:
    """Chat-or-text call that appends an assistant turn to `trace`. Tolerates empty/aborted replies."""
    tag = _pipeline_tag(repo_id)
    msgs_payload = [{"role": t.role, "content": t.content} for t in trace.messages]

    def _mk():
        return InferenceClient(model=repo_id, token=HF_TOKEN, timeout=120)

    def _try_chat(client, mx, temp, sample):
        resp = client.chat_completion(
            messages=msgs_payload,
            max_tokens=mx,
            temperature=temp,
            top_p=0.9 if sample else 1.0,
        )
        if not resp or not getattr(resp, "choices", None):
            raise RuntimeError("empty chat response (no choices)")
        content = resp.choices[0].message.get("content") if hasattr(resp.choices[0], "message") else None
        if not content or not str(content).strip():
            raise RuntimeError("empty chat content")
        return content

    def _try_text(client, mx, temp, sample):
        out = client.text_generation(
            _flatten_messages_for_textgen(msgs_payload),
            max_new_tokens=mx,
            temperature=temp,
            top_p=0.9 if sample else 1.0,
            do_sample=sample,
            repetition_penalty=1.05 if sample else 1.0,
            return_full_text=False,
        )
        text = out if isinstance(out, str) else getattr(out, "generated_text", None)
        if not text or not str(text).strip():
            raise RuntimeError("empty text generation")
        return text

    plans = [
        ("chat", temperature, max_new_tokens, True),
        ("text", temperature, max_new_tokens, True),
        ("chat", 0.0, min(256, max_new_tokens), False),
        ("text", 0.0, min(256, max_new_tokens), False),
    ]

    last_err = None
    for i, (mode, temp, mx, sample) in enumerate(plans):
        try:
            client = _mk()
            if (tag in {"conversational", "chat"}) and mode == "chat":
                text = _try_chat(client, mx, temp, sample)
            elif mode == "text":
                text = _try_text(client, mx, temp, sample)
            else:
                continue
            trace.add("assistant", text, mode=mode, tag=tag, temp=temp, max_tokens=mx, sample=sample)
            return text
        except (StopIteration, Exception) as e:
            last_err = e
            trace.add("assistant", f"[MODEL ERROR] {type(e).__name__}: {e}", mode=mode, tag=tag)
            time.sleep(0.8 + 0.4 * (i + 1))
            continue

    text = "[MODEL ERROR] empty/aborted response after retries"
    trace.add("assistant", text, mode="failed", tag=tag)
    return text

# ---------- Prompt builder (type-aware) ----------
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

    return f"""Issue: {url}
Title: {title}

Body (truncated):
{body_trunc}

TASK:
{task}
Keep it concise and actionable.
"""

# ---------- Generator helpers: create Qs ----------
def generate_questions(gen_model_id: str, issue: dict, itype: IssueType, k: int) -> List[str]:
    """Use a generator LLM to propose k sharp, evaluable questions about THIS issue."""
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()[:2500]
    url = issue.get("html_url", "")

    sys_msg = {"role": "system", "content": "You create precise, evaluable developer questions for triaging PyTorch issues."}
    user_msg = {
        "role": "user",
        "content": (
            f"Issue: {url}\nTitle: {title}\nType: {itype.value}\n\n"
            f"Body (truncated):\n{body}\n\n"
            f"Generate {k} incisive, self-contained questions tailored to THIS issue. "
            "Each question should:\n"
            "- Be specific and actionable\n"
            "- Avoid speculation\n"
            "- Reference PyTorch APIs/files if relevant\n"
            "- Fit on one or two lines\n"
            "Return as a numbered list only."
        )
    }
    raw = _chat_once(gen_model_id, [sys_msg, user_msg], max_new_tokens=500, temperature=0.4)
    lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
    qs = []
    for ln in lines:
        q = re.sub(r"^\s*\d+[\.\)]\s*", "", ln)
        if len(q) > 0:
            qs.append(q)
    out = []
    seen = set()
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= k:
            break
    return out

# ---------- Knowledge Graph helpers ----------
def add_node(nodes: Dict[str, dict], node_id: str, label: str, ntype: str, props: Optional[dict] = None):
    if node_id in nodes:
        return
    nodes[node_id] = {"id": node_id, "label": label, "type": ntype, "props": props or {}}

def add_edge(edges: List[dict], src: str, dst: str, etype: str, props: Optional[dict] = None):
    edges.append({"from": src, "to": dst, "type": etype, "props": props or {}})

# ---------- NEW: Simple stdout conversation printer ----------
def print_conversation_to_stdout(issue_number: int, model_nick: str, kind: str, trace: ChatTrace):
    """
    Minimal pretty-printer for a conversation trace. Prints in the order the turns were added.
    kind: "diagnostic" or f"qna q{index:02d}"
    """
    sep = "=" * 88
    print(sep)
    header = f"Issue #{issue_number} — Model: {model_nick} — {kind}"
    print(header)
    print("-" * len(header))
    for turn in trace.messages:
        role = (turn.role or "unknown").capitalize()
        text = (turn.content or "").strip()
        print(f"{role}: {text}")
    print(sep)
    print()  # blank line for readability

# ---------- Main ----------
def main():
    global SESSION, GH_TIMEOUT

    ap = argparse.ArgumentParser(description="PyTorch issues -> type-aware triage + chat traces + generator Q&A -> KG JSON")
    ap.add_argument("--repo", default="pytorch/pytorch", help="owner/repo (default: pytorch/pytorch)")
    ap.add_argument("--limit", type=int, default=8, help="number of issues to process")
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                    help=f"subset of models (default: {', '.join(MODELS.keys())})")
    ap.add_argument("--sleep", type=float, default=1.0, help="sleep seconds between model calls")

    # Outputs
    ap.add_argument("--out_jsonl", default="runs.jsonl", help="raw per-issue results (JSONL)")
    ap.add_argument("--out_kg", default="knowledge_graph.json", help="knowledge graph output (JSON)")

    # Filters / behavior
    ap.add_argument("--include-types", nargs="*", default=["bug", "feature", "question", "docs", "discussion", "task", "other"],
                    help="Process only these issue types (default: all).")
    ap.add_argument("--skip-types", nargs="*", default=[],
                    help="Skip these issue types (e.g., --skip-types feature).")
    ap.add_argument("--turns", type=int, default=2,
                    help="Assistant conversation turns per model for diagnostic run (default: 2).")

    # GitHub HTTP robustness knobs
    ap.add_argument("--gh-timeout", type=float, default=45.0, help="GitHub request timeout (seconds)")
    ap.add_argument("--gh-retries", type=int, default=5, help="Max retries for GitHub requests")
    ap.add_argument("--gh-backoff", type=float, default=0.6, help="Exponential backoff factor for GitHub retries")

    # Prompt-generator driven Q&A
    ap.add_argument(
        "--gen-model",
        default=GEN_MODEL[0],  # default to the repo_id from GEN_MODEL
        help="HF model id used to generate questions/prompts (default: GEN_MODEL)."
    )
    ap.add_argument("--qna-per-issue", type=int, default=3,
                    help="Number of Q&A pairs to generate per issue.")
    ap.add_argument("--qna-turns", type=int, default=1,
                    help="Assistant turns per question per model (in addition to the initial answer).")

    args = ap.parse_args()

    GH_TIMEOUT = args.gh_timeout
    SESSION = _requests_session(retries=args.gh_retries, backoff=args.gh_backoff)

    issues = fetch_issues(args.repo, args.limit, state="all")
    print(f"Fetched {len(issues)} issues from {args.repo}")

    nodes: Dict[str, dict] = {}
    edges: List[dict] = []

    # Schema (for reference in output)
    schema = {
        "node_types": [
            "Model", "Issue", "Solution", "RootCause",
            "Conversation", "Turn",
            "Question", "Answer"
        ],
        "edge_types": [
            "answered", "produced", "about", "has_turn", "next",
            "solves", "implements", "causes", "proposes",
            "has_q", "has_a", "answered_by"
        ]
    }

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

        itype = classify_issue(title, labels, body)
        if itype.value not in set(args.include_types) or itype.value in set(args.skip_types):
            print(f"Skipping issue #{number} (type={itype.value}) due to filters.")
            continue

        print(f"\n→ Issue #{number} [{itype.value}]: {title}")
        comments = fetch_issue_comments(args.repo, number)

        fixed = is_fixed(ishu, comments, itype)
        solution_text, rca_text = extract_solution_and_root_cause(comments)

        if fixed and not solution_text:
            pr_num = find_closing_pr(args.repo, number)
            if pr_num:
                pr_body, commit_msgs = fetch_pr_body_and_commits(args.repo, pr_num)
                pr_solution, pr_rca = extract_solution_rca_from_texts([pr_body] + commit_msgs)
                if pr_solution:
                    solution_text = pr_solution
                if pr_rca and not rca_text:
                    rca_text = pr_rca

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
        if solution_text:
            solution_node_id = f"solution::{args.repo}#{number}"
            add_node(nodes, solution_node_id, f"Solution for {args.repo}#{number}", "Solution",
                     {"text": solution_text})
            edge_type = "implements" if itype == IssueType.FEATURE else "solves"
            add_edge(edges, solution_node_id, issue_node_id, edge_type)
        elif fixed:
            solution_node_id = f"solution::{args.repo}#{number}"
            add_node(nodes, solution_node_id, f"Solution for {args.repo}#{number}", "Solution",
                     {"text": "Solution likely in linked PR/commit/comments; needs manual curation."})
            edge_type = "implements" if itype == IssueType.FEATURE else "solves"
            add_edge(edges, solution_node_id, issue_node_id, edge_type)

        if rca_text:
            rca_node_id = f"rca::{args.repo}#{number}"
            add_node(nodes, rca_node_id, f"RCA for {args.repo}#{number}", "RootCause", {"text": rca_text})
            add_edge(edges, rca_node_id, issue_node_id, "causes")

        # === Diagnostic conversation per model (2-turn default) ===
        system_prompt = "You are a concise, accurate PyTorch triage assistant."
        issue_prompt = build_prompt(ishu, itype)

        diag_model_records = []
        for nick in args.models:
            repo_id, family, size = MODELS[nick]
            trace = ChatTrace(model_id=repo_id, model_nick=nick, family=family, size=size, messages=[])
            trace.add("system", system_prompt)
            trace.add("user", issue_prompt)

            _ = call_hf_chat(repo_id, trace)

            if args.turns >= 2:
                followup = (
                    "Summarize strictly:\n"
                    f"- Type ({itype.value})\n"
                    "- Root cause (1 sentence)\n"
                    "- Minimal repro (bullets)\n"
                    "- Fix/Workaround (bullets)\n"
                    "- References (links or file paths)"
                )
                trace.add("user", followup)
                _ = call_hf_chat(repo_id, trace)

            for _extra in range(max(0, args.turns - 2)):
                trace.add("user", "Refine your answer. Keep it shorter and more actionable.")
                _ = call_hf_chat(repo_id, trace)

            # >>> NEW: print diagnostic conversation to stdout
            print_conversation_to_stdout(number, nick, "diagnostic", trace)

            # Knowledge graph: Conversation + Turn nodes (diagnostic)
            conv_id = f"conv::{nick}::{args.repo}#{number}"
            add_node(nodes, conv_id, f"Conversation {nick} #{number}", "Conversation", {
                "model": nick, "family": family, "size": size, "kind": "diagnostic"
            })
            add_edge(edges, f"model::{family}::{size}", conv_id, "produced")
            add_edge(edges, conv_id, issue_node_id, "about")

            prev_turn_id = None
            for i, t in enumerate(trace.messages):
                turn_id = f"{conv_id}::t{i:02d}"
                add_node(nodes, turn_id, f"{t.role} t{i}", "Turn", {
                    "role": t.role,
                    "content": t.content,
                    "ts": t.ts,
                    **(t.meta or {})
                })
                add_edge(edges, conv_id, turn_id, "has_turn")
                if prev_turn_id:
                    add_edge(edges, prev_turn_id, turn_id, "next")
                prev_turn_id = turn_id

            if itype == IssueType.FEATURE and not solution_text:
                add_edge(edges, conv_id, issue_node_id, "proposes")

            diag_model_records.append({
                "nickname": nick,
                "family": family,
                "param_size": size,
                "hf_repo": repo_id,
                "trace": [asdict(m) for m in trace.messages]
            })

            time.sleep(args.sleep)

        # === Generator-driven Q&A pairs ===
        q_list = generate_questions(args.gen_model, ishu, itype, args.qna_per_issue)

        qna_model_records = []
        for q_index, question_text in enumerate(q_list, start=1):
            for nick in args.models:
                repo_id, family, size = MODELS[nick]

                conv_id = f"qna::{nick}::{args.repo}#{number}::q{q_index:02d}"
                trace = ChatTrace(model_id=repo_id, model_nick=nick, family=family, size=size, messages=[])
                trace.add("system", system_prompt)
                context = f"(Context: {issue_url})\n"
                trace.add("user", context + question_text)

                answer_text = call_hf_chat(repo_id, trace)

                for _turn in range(max(0, args.qna_turns - 1)):
                    trace.add("user", "Refine answer in 3-6 bullet points with code or links if helpful.")
                    answer_text = call_hf_chat(repo_id, trace)

                # >>> NEW: print QnA conversation to stdout
                print_conversation_to_stdout(number, nick, f"qna q{q_index:02d}", trace)

                # Conversation + Turns for QnA
                add_node(nodes, conv_id, f"QnA {nick} q{q_index}", "Conversation", {
                    "model": nick, "family": family, "size": size, "kind": "qna", "q_index": q_index
                })
                add_edge(edges, f"model::{family}::{size}", conv_id, "produced")
                add_edge(edges, conv_id, issue_node_id, "about")

                prev_turn_id = None
                for i, t in enumerate(trace.messages):
                    turn_id = f"{conv_id}::t{i:02d}"
                    add_node(nodes, turn_id, f"{t.role} t{i}", "Turn", {
                        "role": t.role, "content": t.content, "ts": t.ts, **(t.meta or {})
                    })
                    add_edge(edges, conv_id, turn_id, "has_turn")
                    if prev_turn_id:
                        add_edge(edges, prev_turn_id, turn_id, "next")
                    prev_turn_id = turn_id

                # Explicit Q & A nodes
                qa_id = f"qa::{nick}::{args.repo}#{number}::q{q_index:02d}"
                q_node = qa_id + "::Q"
                a_node = qa_id + "::A"

                add_node(nodes, q_node, f"Q{q_index}", "Question", {
                    "text": question_text, "gen_model": args.gen_model
                })
                add_node(nodes, a_node, f"A{q_index} ({nick})", "Answer", {
                    "text": answer_text, "model": nick, "family": family, "size": size
                })

                add_edge(edges, q_node, a_node, "answered_by")
                add_edge(edges, conv_id, q_node, "has_q")
                add_edge(edges, conv_id, a_node, "has_a")
                add_edge(edges, q_node, issue_node_id, "about")
                add_edge(edges, a_node, issue_node_id, "about")

                qna_model_records.append({
                    "q_index": q_index,
                    "question": question_text,
                    "model": nick,
                    "family": family,
                    "param_size": size,
                    "hf_repo": repo_id,
                    "trace": [asdict(m) for m in trace.messages],
                    "final_answer": answer_text
                })

                time.sleep(args.sleep)

        # Raw record per issue (JSONL)
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
            "diagnostic_models": diag_model_records,
            "qna_models": qna_model_records
        }
        jl.write(json.dumps(record, ensure_ascii=False) + "\n")
        jl.flush()

    jl.close()

    # Emit Knowledge Graph JSON
    kg = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": schema,
        "nodes": list(nodes.values()),
        "edges": edges
    }
    with open(args.out_kg, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    print(f"\nDone.\n- Raw per-issue runs: {args.out_jsonl}\n- Knowledge graph JSON: {args.out_kg}")

if __name__ == "__main__":
    main()