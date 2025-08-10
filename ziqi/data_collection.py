import os
import time
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Optional

import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)
GH_TOKEN = os.getenv("GITHUB_TOKEN", None)

# ---------- Models (small, free-tier friendly) ----------
MODELS = {
    "deepseek-coder-1.3b": ("deepseek-ai/deepseek-coder-1.3b-instruct", "DeepSeek-Coder", "1.3B"),
    "deepseek-coder-6.7b": ("deepseek-ai/deepseek-coder-6.7b-instruct", "DeepSeek-Coder", "6.7B"),
    "qwen2.5-coder-1.5b":  ("Qwen/Qwen2.5-Coder-1.5B-Instruct",        "Qwen2.5-Coder",   "1.5B"),
    "starcoder2-3b":       ("bigcode/starcoder2-3b",                    "StarCoder2",      "3B"),
    "phi-3-mini":          ("microsoft/Phi-3-mini-128k-instruct",       "Phi-3",           "3.8B"),
}

# ---------- GitHub helpers ----------
GH_API = "https://api.github.com"
GH_HEADERS = {"Accept": "application/vnd.github+json"}
if GH_TOKEN:
    GH_HEADERS["Authorization"] = f"Bearer {GH_TOKEN}"

def gh_get(url, params=None):
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
    out = []
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
        time.sleep(0.3)
    return out

def fetch_issue_comments(repo: str, number: int) -> List[dict]:
    url = f"{GH_API}/repos/{repo}/issues/{number}/comments"
    comments = []
    page = 1
    while True:
        batch = gh_get(url, {"page": page, "per_page": 100})
        if not batch:
            break
        comments.extend(batch)
        page += 1
        time.sleep(0.2)
    return comments

# ---------- Fixed/solution heuristics ----------
FIX_PHRASES = [
    "fixed by", "closed by", "resolved in", "landed in", "merged in",
    "this was fixed", "should be fixed", "will be fixed"
]
SOLUTION_HOOKS = [
    "workaround", "solution", "fix", "patch", "proposed change",
    "apply this", "use this", "add this", "upgrade to", "revert"
]

def is_fixed(issue: dict, comments: List[dict]) -> bool:
    """Heuristic: closed + maintainer mentions fix/merge OR body/comments contain fix phrases."""
    if issue.get("state") == "closed":
        return True
    text_blobs = [issue.get("title") or "", issue.get("body") or ""]
    text_blobs += [c.get("body") or "" for c in comments]
    joined = "\n".join(text_blobs).lower()
    return any(p in joined for p in FIX_PHRASES)

def extract_solution_and_root_cause(comments: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Pull a likely solution and RCA from maintainer/collaborator comments.
    Very heuristic: prefer comments from project members that contain code blocks or solution hooks.
    """
    maintainer_roles = {"MEMBER", "OWNER", "COLLABORATOR"}
    candidates = []
    for c in comments:
        role = (c.get("author_association") or "").upper()
        body = (c.get("body") or "").strip()
        if role in maintainer_roles and len(body) > 30:
            score = 0
            if "```" in body or "~~~" in body: score += 2
            if any(h in body.lower() for h in SOLUTION_HOOKS): score += 2
            if len(body) > 200: score += 1
            candidates.append((score, body))

    candidates.sort(reverse=True, key=lambda x: x[0])
    solution = candidates[0][1] if candidates else None

    # crude RCA extraction: first sentence that looks like cause
    rca = None
    if solution:
        lines = re.split(r"[.\n]", solution)
        for ln in lines:
            l = ln.strip().lower()
            if any(k in l for k in ["because", "due to", "caused by", "root cause", "reason is", "it happens when"]):
                rca = ln.strip()
                break
    return solution, rca

# ---------- LLM calls ----------
def build_prompt(issue: dict) -> str:
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()
    url = issue.get("html_url", "")
    body_trunc = body[:3000]
    return f"""You are triaging a PyTorch GitHub issue.

Issue: {url}
Title: {title}

Body (truncated):
{body_trunc}

Please provide:
1) Likely root cause (usage vs library bug vs environment/build), with justification.
2) Steps to reproduce or inspect.
3) Proposed fix or workaround (code or config).
4) References to relevant PyTorch docs/files (best effort).
Keep it concise and actionable.
"""

def call_hf(repo_id: str, prompt: str, temperature=0.2, max_new_tokens=500) -> str:
    client = InferenceClient(model=repo_id, token=HF_TOKEN, timeout=120)
    try:
        out = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            return_full_text=False,
        )
        return out if isinstance(out, str) else getattr(out, "generated_text", str(out))
    except Exception as e:
        return f"[MODEL ERROR] {type(e).__name__}: {e}"

# ---------- Knowledge Graph builder ----------
def add_node(nodes, node_id, label, ntype, props=None):
    if node_id in nodes:
        return
    nodes[node_id] = {"id": node_id, "label": label, "type": ntype, "props": props or {}}

def add_edge(edges, src, dst, etype, props=None):
    edges.append({"from": src, "to": dst, "type": etype, "props": props or {}})

def main():
    ap = argparse.ArgumentParser(description="PyTorch issue scraper -> solution/rca + small-model answers -> KG JSON")
    ap.add_argument("--repo", default="pytorch/pytorch", help="owner/repo")
    ap.add_argument("--limit", type=int, default=8, help="number of issues to process")
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()),
                    help=f"subset of models (default: {', '.join(MODELS.keys())})")
    ap.add_argument("--sleep", type=float, default=1.0, help="sleep seconds between model calls")
    ap.add_argument("--out_jsonl", default="runs.jsonl", help="raw per-issue results")
    ap.add_argument("--out_kg", default="knowledge_graph.json", help="knowledge graph output")
    args = ap.parse_args()

    issues = fetch_issues(args.repo, args.limit, state="all")
    print(f"Fetched {len(issues)} issues from {args.repo}")

    nodes = {}   # id -> node
    edges = []   # list of edges

    # seed model family nodes
    for nick in args.models:
        repo_id, family, size = MODELS[nick]
        model_node_id = f"model::{family}::{size}"
        add_node(nodes, model_node_id, f"{family} {size}", "Model", {"nickname": nick, "hf_repo": repo_id})

    jl = open(args.out_jsonl, "a", encoding="utf-8")

    for ishu in issues:
        number = ishu["number"]
        issue_url = ishu["html_url"]
        print(f"\n→ Issue #{number}: {issue_url}")

        comments = fetch_issue_comments(args.repo, number)
        fixed = is_fixed(ishu, comments)
        solution_text, rca_text = extract_solution_and_root_cause(comments)

        # Create issue node
        issue_node_id = f"issue::{args.repo}#{number}"
        add_node(nodes, issue_node_id, f"{args.repo}#{number}", "Issue", {
            "title": ishu.get("title",""),
            "url": issue_url,
            "state": ishu.get("state",""),
            "created_at": ishu.get("created_at",""),
            "updated_at": ishu.get("updated_at",""),
            "labels": [l["name"] for l in ishu.get("labels",[])]
        })

        # If we found solution/rca, attach nodes
        solution_node_id = None
        rca_node_id = None
        if solution_text:
            solution_node_id = f"solution::{args.repo}#{number}"
            add_node(nodes, solution_node_id, f"Solution for {args.repo}#{number}", "Solution", {"text": solution_text})
            add_edge(edges, solution_node_id, issue_node_id, "solves")
        if rca_text:
            rca_node_id = f"rca::{args.repo}#{number}"
            add_node(nodes, rca_node_id, f"RCA for {args.repo}#{number}", "RootCause", {"text": rca_text})
            add_edge(edges, rca_node_id, issue_node_id, "causes")

        # Run small models (always, or you could guard with `if not fixed`)
        prompt = build_prompt(ishu)
        model_answers = {}
        for nick in args.models:
            repo_id, family, size = MODELS[nick]
            ans = call_hf(repo_id, prompt)
            model_answers[nick] = ans
            # Connect model to issue in KG
            model_node_id = f"model::{family}::{size}"
            add_edge(edges, model_node_id, issue_node_id, "answered")
            # Optionally, add a node for this specific answer artifact
            ans_node_id = f"answer::{nick}::{args.repo}#{number}"
            add_node(nodes, ans_node_id, f"Answer by {nick} on #{number}", "Answer", {"text": ans})
            add_edge(edges, model_node_id, ans_node_id, "produced")
            add_edge(edges, ans_node_id, issue_node_id, "about")
            time.sleep(args.sleep)

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "repo": args.repo,
            "issue_number": number,
            "issue_url": issue_url,
            "title": ishu.get("title",""),
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

        # If marked fixed but no solution found, add a placeholder Solution node to be curated later
        if fixed and not solution_text:
            placeholder = f"Solution likely in linked PR/commit/comments; manual curation needed."
            solution_node_id = solution_node_id or f"solution::{args.repo}#{number}"
            if solution_node_id not in nodes:
                add_node(nodes, solution_node_id, f"Solution for {args.repo}#{number}", "Solution", {"text": placeholder})
                add_edge(edges, solution_node_id, issue_node_id, "solves")

    jl.close()

    kg = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema": {
            "node_types": ["Model", "Issue", "Solution", "RootCause", "Answer"],
            "edge_types": ["answered", "produced", "about", "solves", "causes"]
        },
        "nodes": list(nodes.values()),
        "edges": edges
    }
    with open(args.out_kg, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    print(f"\nDone.\n- Raw per-issue runs: {args.out_jsonl}\n- Knowledge graph JSON: {args.out_kg}")
    print("Open the KG JSON in a graph tool (Gephi/Cytoscape) or load in Python to visualize.")

if __name__ == "__main__":
    main()