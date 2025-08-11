#!/usr/bin/env python3

"""
Export the Kùzu database in `overfit-db` to a JSON file compatible with the frontend graph.

Produces an object with two arrays: `nodes` and `links` where nodes have at least
{id, type, label} and links have at least {source, target, type}. Optional attributes like
`strength`, `reason`, or timestamps are included when available.

Usage:
  python3 export_to_graph_json.py \
    --db /absolute/path/to/OverFit/kuzu/overfit-db \
    --out /absolute/path/to/OverFit/frontend/assets/graph.json

Requires:
  pip install kuzu
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import kuzu
except Exception as exc:
    print("Error: Python package 'kuzu' is required. Install with: pip install kuzu", file=sys.stderr)
    raise


def parse_args():
    parser = argparse.ArgumentParser(description="Export Kùzu graph to frontend JSON")
    parser.add_argument("--db", required=True, help="Absolute path to Kùzu database directory (e.g., .../kuzu/overfit-db)")
    parser.add_argument("--out", required=True, help="Absolute path to output JSON file (e.g., .../frontend/assets/graph.json)")
    return parser.parse_args()


def connect_database(db_path):
    if not os.path.isdir(db_path):
        raise FileNotFoundError(f"Database dir does not exist: {db_path}")
    database = kuzu.Database(db_path)
    connection = kuzu.Connection(database)
    return connection


def fetch_rows(connection, query):
    result = connection.execute(query)
    rows = []
    while result.has_next():
        row = result.get_next()
        rows.append(row)
    return rows, result.get_column_names()


def export_nodes(connection):
    nodes_by_id = {}  # type: Dict[str, Dict]

    def add_node(node_id, node_type, label, extra=None):
        if node_id in nodes_by_id:
            return
        node = {"id": node_id, "type": node_type, "label": label}
        if extra:
            node.update(extra)
        nodes_by_id[node_id] = node

    # Problem
    rows, cols = fetch_rows(connection, """
        MATCH (n:Problem)
        RETURN n.id AS id, n.text AS text, n.type AS problem_type, n.variables AS variables, n.metadata AS metadata
    """)
    for row in rows:
        node_id = row[0]
        text = row[1] or node_id
        problem_type = row[2]
        variables = row[3]
        metadata = row[4]
        add_node(
            node_id=node_id,
            node_type="problem",
            label=text,
            extra={
                "problem_type": problem_type,
                "variables": variables,
                "metadata": safe_parse_json(metadata),
            },
        )

    # Solution
    rows, cols = fetch_rows(connection, """
        MATCH (n:Solution)
        RETURN n.id AS id, n.problem_id AS problem_id, n.expression AS expression, n.verified AS verified, n.metadata AS metadata
    """)
    for row in rows:
        node_id = row[0]
        problem_id = row[1]
        expression = row[2] or node_id
        verified = bool(row[3]) if row[3] is not None else None
        metadata = row[4]
        add_node(
            node_id=node_id,
            node_type="solution",
            label=expression,
            extra={
                "problem_id": problem_id,
                "verified": verified,
                "metadata": safe_parse_json(metadata),
            },
        )

    # Method
    rows, cols = fetch_rows(connection, """
        MATCH (n:Method)
        RETURN n.id AS id, n.name AS name, n.preconditions AS preconditions, n.metadata AS metadata
    """)
    for row in rows:
        node_id = row[0]
        name = row[1] or node_id
        preconditions = row[2]
        metadata = row[3]
        add_node(
            node_id=node_id,
            node_type="method",
            label=name,
            extra={
                "preconditions": preconditions,
                "metadata": safe_parse_json(metadata),
            },
        )

    # Concept
    rows, cols = fetch_rows(connection, """
        MATCH (n:Concept)
        RETURN n.id AS id, n.name AS name, n.definition AS definition, n.metadata AS metadata
    """)
    for row in rows:
        node_id = row[0]
        name = row[1] or node_id
        definition = row[2]
        metadata = row[3]
        add_node(
            node_id=node_id,
            node_type="concept",
            label=name,
            extra={
                "definition": definition,
                "metadata": safe_parse_json(metadata),
            },
        )

    # Model
    rows, cols = fetch_rows(connection, """
        MATCH (n:Model)
        RETURN n.id AS id, n.name AS name, n.metadata AS metadata
    """)
    for row in rows:
        node_id = row[0]
        name = row[1] or node_id
        metadata = row[2]
        add_node(
            node_id=node_id,
            node_type="model",
            label=name,
            extra={
                "metadata": safe_parse_json(metadata),
            },
        )

    return nodes_by_id


def safe_parse_json(value):
    if value is None:
        return None
    if isinstance(value, dict) or isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def export_links(connection):
    links = []  # type: List[Dict]

    def add_link(source, target, link_type, extra=None):
        link = {"source": source, "target": target, "type": link_type}
        if extra:
            link.update(extra)
        links.append(link)

    # SOLVES (Solution -> Problem)
    rows, cols = fetch_rows(connection, """
        MATCH (s:Solution)-[r:SOLVES]->(p:Problem)
        RETURN s.id AS sid, p.id AS pid, r.created_at AS created_at
    """)
    for row in rows:
        sid = row[0]
        pid = row[1]
        created_at = row[2]
        add_link(sid, pid, "solves", {"created_at": created_at})

    # RELATED_TO (Problem -> Problem) as similarity
    rows, cols = fetch_rows(connection, """
        MATCH (a:Problem)-[r:RELATED_TO]->(b:Problem)
        RETURN a.id AS aid, b.id AS bid, r.similarity AS similarity
    """)
    for row in rows:
        aid = row[0]
        bid = row[1]
        sim = row[2]
        add_link(aid, bid, "similarity", {"strength": sim})

    # SIMILAR_TO (Solution -> Solution) as similarity
    rows, cols = fetch_rows(connection, """
        MATCH (a:Solution)-[r:SIMILAR_TO]->(b:Solution)
        RETURN a.id AS aid, b.id AS bid, r.similarity AS similarity
    """)
    for row in rows:
        aid = row[0]
        bid = row[1]
        sim = row[2]
        add_link(aid, bid, "similarity", {"strength": sim})

    # CAN_SOLVE (Model -> Problem) as related with strength
    rows, cols = fetch_rows(connection, """
        MATCH (m:Model)-[r:CAN_SOLVE]->(p:Problem)
        RETURN m.id AS mid, p.id AS pid, r.confidence AS confidence
    """)
    can_solve_by_problem = defaultdict(list)  # type: Dict[str, List[Tuple[str, float]]]
    for row in rows:
        mid = row[0]
        pid = row[1]
        conf = float(row[2]) if row[2] is not None else None
        add_link(mid, pid, "related", {"strength": conf, "role": "can_solve"})
        if conf is not None:
            can_solve_by_problem[pid].append((mid, conf))

    # CANT_SOLVE (Model -> Problem) as related with reason
    rows, cols = fetch_rows(connection, """
        MATCH (m:Model)-[r:CANT_SOLVE]->(p:Problem)
        RETURN m.id AS mid, p.id AS pid, r.reason AS reason
    """)
    for row in rows:
        mid = row[0]
        pid = row[1]
        reason = row[2]
        add_link(mid, pid, "related", {"reason": reason, "role": "cant_solve"})

    # ABOUT (Problem -> Concept)
    rows, cols = fetch_rows(connection, """
        MATCH (p:Problem)-[:ABOUT]->(c:Concept)
        RETURN p.id AS pid, c.id AS cid
    """)
    for row in rows:
        pid = row[0]
        cid = row[1]
        add_link(pid, cid, "related", {"role": "about"})

    # APPLIES_TO (Method -> Problem)
    rows, cols = fetch_rows(connection, """
        MATCH (m:Method)-[:APPLIES_TO]->(p:Problem)
        RETURN m.id AS mid, p.id AS pid
    """)
    for row in rows:
        mid = row[0]
        pid = row[1]
        add_link(mid, pid, "related", {"role": "applies_to"})

    # USES (Method -> Concept)
    rows, cols = fetch_rows(connection, """
        MATCH (m:Method)-[:USES]->(c:Concept)
        RETURN m.id AS mid, c.id AS cid
    """)
    for row in rows:
        mid = row[0]
        cid = row[1]
        add_link(mid, cid, "related", {"role": "uses"})

    # REFERENCES (Solution -> Concept)
    rows, cols = fetch_rows(connection, """
        MATCH (s:Solution)-[:REFERENCES]->(c:Concept)
        RETURN s.id AS sid, c.id AS cid
    """)
    for row in rows:
        sid = row[0]
        cid = row[1]
        add_link(sid, cid, "related", {"role": "references"})

    # DEPENDS_ON (Concept -> Concept)
    rows, cols = fetch_rows(connection, """
        MATCH (a:Concept)-[:DEPENDS_ON]->(b:Concept)
        RETURN a.id AS aid, b.id AS bid
    """)
    for row in rows:
        aid = row[0]
        bid = row[1]
        add_link(aid, bid, "related", {"role": "depends_on"})

    # CAUSED_BY (Problem -> Problem)
    rows, cols = fetch_rows(connection, """
        MATCH (a:Problem)-[:CAUSED_BY]->(b:Problem)
        RETURN a.id AS aid, b.id AS bid
    """)
    for row in rows:
        aid = row[0]
        bid = row[1]
        add_link(aid, bid, "related", {"role": "caused_by"})

    # WORKAROUND_FOR (Method -> Problem)
    rows, cols = fetch_rows(connection, """
        MATCH (m:Method)-[:WORKAROUND_FOR]->(p:Problem)
        RETURN m.id AS mid, p.id AS pid
    """)
    for row in rows:
        mid = row[0]
        pid = row[1]
        add_link(mid, pid, "related", {"role": "workaround_for"})

    # Synthetic PRODUCED_BY (Solution -> Model): choose highest-confidence model for the solved problem
    rows, cols = fetch_rows(connection, """
        MATCH (s:Solution)-[:SOLVES]->(p:Problem)
        RETURN s.id AS sid, p.id AS pid
    """)
    best_model_for_problem = {}  # type: Dict[str, str]
    for pid, pairs in can_solve_by_problem.items():
        if not pairs:
            continue
        pairs_sorted = sorted(pairs, key=lambda x: (x[1] if x[1] is not None else -1.0), reverse=True)
        best_mid = pairs_sorted[0][0]
        best_model_for_problem[pid] = best_mid

    for row in rows:
        sid = row[0]
        pid = row[1]
        mid = best_model_for_problem.get(pid)
        if mid:
            add_link(sid, mid, "produced_by")

    return links


def main():
    args = parse_args()
    connection = connect_database(args.db)

    nodes_by_id = export_nodes(connection)
    links = export_links(connection)

    nodes = list(nodes_by_id.values())

    graph = {"nodes": nodes, "links": links}

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)

    print(f"Wrote graph JSON: {args.out}")


if __name__ == "__main__":
    main() 