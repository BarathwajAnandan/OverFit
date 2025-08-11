#!/usr/bin/env python3
"""
Export graph data from KuzuDB to JSON for frontend visualization.
"""

import kuzu
import json
import os
import sys
from datetime import datetime

def connect_to_db(db_path):
    """Connect to KuzuDB database."""
    try:
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def query_nodes(conn):
    """Query all node tables and return unified node data."""
    nodes = []
    
    # Query Problems
    try:
        result = conn.execute("MATCH (p:Problem) RETURN p.id, p.text, p.type, p.variables, p.metadata")
        while result.has_next():
            row = result.get_next()
            nodes.append({
                'id': row[0],
                'type': 'problem',
                'label': row[1][:50] + '...' if row[1] and len(row[1]) > 50 else row[1],
                'text': row[1],
                'problem_type': row[2],
                'variables': row[3],
                'metadata': row[4],
                'karma': 5 + len(row[0]) % 15  # Generate some karma based on ID
            })
    except Exception as e:
        print(f"Error querying Problems: {e}")
    
    # Query Solutions
    try:
        result = conn.execute("MATCH (s:Solution) RETURN s.id, s.problem_id, s.expression, s.verified, s.metadata")
        while result.has_next():
            row = result.get_next()
            nodes.append({
                'id': row[0],
                'type': 'solution',
                'label': row[2][:50] + '...' if row[2] and len(row[2]) > 50 else row[2],
                'expression': row[2],
                'problem_id': row[1],
                'verified': row[3],
                'metadata': row[4],
                'reuseScore': 0.6 + (len(row[0]) % 40) / 100,  # Generate reuse score
                'upvotes': 10 + len(row[0]) % 25
            })
    except Exception as e:
        print(f"Error querying Solutions: {e}")
    
    # Query Models
    try:
        result = conn.execute("MATCH (m:Model) RETURN m.id, m.name, m.metadata")
        while result.has_next():
            row = result.get_next()
            nodes.append({
                'id': row[0],
                'type': 'model',
                'label': row[1],
                'name': row[1],
                'metadata': row[2],
                'size': 'Large' if '70B' in row[1] or 'GPT-4' in row[1] else 'Medium' if 'Qwen' in row[1] else 'Small',
                'provider': 'Meta' if 'Llama' in row[1] else 'OpenAI' if 'GPT' in row[1] else 'Microsoft' if 'DeepSeek' in row[1] else 'Other'
            })
    except Exception as e:
        print(f"Error querying Models: {e}")
    
    # Query Concepts
    try:
        result = conn.execute("MATCH (c:Concept) RETURN c.id, c.name, c.definition, c.metadata")
        while result.has_next():
            row = result.get_next()
            nodes.append({
                'id': row[0],
                'type': 'concept',
                'label': row[1],
                'name': row[1],
                'definition': row[2],
                'metadata': row[3]
            })
    except Exception as e:
        print(f"Error querying Concepts: {e}")
    
    # Query Methods
    try:
        result = conn.execute("MATCH (m:Method) RETURN m.id, m.name, m.preconditions, m.metadata")
        while result.has_next():
            row = result.get_next()
            nodes.append({
                'id': row[0],
                'type': 'method',
                'label': row[1],
                'name': row[1],
                'preconditions': row[2],
                'metadata': row[3]
            })
    except Exception as e:
        print(f"Error querying Methods: {e}")
    
    return nodes

def query_links(conn):
    """Query all relationship tables and return unified link data."""
    links = []
    
    # Query SOLVES relationships
    try:
        result = conn.execute("MATCH (s:Solution)-[r:SOLVES]->(p:Problem) RETURN s.id, p.id, r.created_at")
        while result.has_next():
            row = result.get_next()
            links.append({
                'source': row[0],
                'target': row[1],
                'type': 'solves',
                'strength': 1.0,
                'created_at': str(row[2]) if row[2] else None
            })
    except Exception as e:
        print(f"Error querying SOLVES: {e}")
    
    # Query CAN_SOLVE relationships
    try:
        result = conn.execute("MATCH (m:Model)-[r:CAN_SOLVE]->(p:Problem) RETURN m.id, p.id, r.confidence")
        while result.has_next():
            row = result.get_next()
            links.append({
                'source': row[0],
                'target': row[1],
                'type': 'can_solve',
                'strength': row[2] if row[2] else 0.5,
                'confidence': row[2]
            })
    except Exception as e:
        print(f"Error querying CAN_SOLVE: {e}")
    
    # Query ABOUT relationships
    try:
        result = conn.execute("MATCH (p:Problem)-[r:ABOUT]->(c:Concept) RETURN p.id, c.id")
        while result.has_next():
            row = result.get_next()
            links.append({
                'source': row[0],
                'target': row[1],
                'type': 'about',
                'strength': 0.7
            })
    except Exception as e:
        print(f"Error querying ABOUT: {e}")
    
    # Query USES relationships
    try:
        result = conn.execute("MATCH (m:Method)-[r:USES]->(c:Concept) RETURN m.id, c.id")
        while result.has_next():
            row = result.get_next()
            links.append({
                'source': row[0],
                'target': row[1],
                'type': 'uses',
                'strength': 0.6
            })
    except Exception as e:
        print(f"Error querying USES: {e}")
    
    # Query REFERENCES relationships
    try:
        result = conn.execute("MATCH (s:Solution)-[r:REFERENCES]->(c:Concept) RETURN s.id, c.id")
        while result.has_next():
            row = result.get_next()
            links.append({
                'source': row[0],
                'target': row[1],
                'type': 'references',
                'strength': 0.5
            })
    except Exception as e:
        print(f"Error querying REFERENCES: {e}")
    
    # Query SIMILAR_TO relationships
    try:
        result = conn.execute("MATCH (s1:Solution)-[r:SIMILAR_TO]->(s2:Solution) RETURN s1.id, s2.id, r.similarity")
        while result.has_next():
            row = result.get_next()
            links.append({
                'source': row[0],
                'target': row[1],
                'type': 'similar_to',
                'strength': row[2] if row[2] else 0.5,
                'similarity': row[2]
            })
    except Exception as e:
        print(f"Error querying SIMILAR_TO: {e}")
    
    return links

def export_to_json(nodes, links, output_path):
    """Export nodes and links to JSON file."""
    graph_data = {
        'nodes': nodes,
        'links': links,
        'exported_at': datetime.now().isoformat(),
        'node_count': len(nodes),
        'link_count': len(links)
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"Graph data exported to {output_path}")
        print(f"Nodes: {len(nodes)}, Links: {len(links)}")
        return True
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        return False

def main():
    db_path = "kuzu/overfit-db"
    output_path = "frontend/assets/graph_data.json"
    
    if not os.path.exists(db_path):
        print(f"Database path {db_path} does not exist!")
        sys.exit(1)
    
    print("Connecting to KuzuDB...")
    conn = connect_to_db(db_path)
    if not conn:
        sys.exit(1)
    
    print("Querying nodes...")
    nodes = query_nodes(conn)
    
    print("Querying relationships...")
    links = query_links(conn)
    
    print("Exporting to JSON...")
    if export_to_json(nodes, links, output_path):
        print("Export completed successfully!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 