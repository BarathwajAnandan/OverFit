from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import uvicorn
import sqlite3
from datetime import datetime
import kuzu
import openai
import os
import json

app = FastAPI(title="Agent Marketplace API")

# Database setup
DB_PATH = "overfit.db"

def init_database():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create registrations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS registrations (
            uuid TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_parameters TEXT,
            seed_count INTEGER DEFAULT 0,
            leech_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get a database connection"""
    return sqlite3.connect(DB_PATH)

# Initialize database on startup
init_database()

# OpenAI Configuration
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_cypher_query(user_input: str) -> str:
    """
    Convert natural language question into Kuzu Cypher query using OpenAI
    """
    # Define the schema for the knowledge graph
    schema_info = """
    Schema Information for Kuzu Database:
    
    Node Types:
    - Problem(id: STRING, text: STRING, type: STRING) - Represents coding problems/issues
    - Concept(id: STRING) - Represents programming concepts, technologies, frameworks
    
    Relationship Types:
    - (Problem)-[:ABOUT]->(Concept) - Problem is related to a specific concept
    
    Available Problem Types: compilation, debugging, compatibility, code_quality, setup, optimization
    Available Concept IDs: cuda, pytorch, python, javascript, react, etc.
    """
    
    kuzu_rules = """
    Kuzu Cypher Rules:
    1. Always specify node and relationship labels explicitly
    2. Relationship cannot be omitted - use -[]- instead of --
    3. Use CONTAINS for partial string matching
    4. Use LIMIT to restrict results
    5. Property access uses dot notation: n.property
    """
    
    prompt = f"""{schema_info}

{kuzu_rules}

Convert the following natural language question into a Kuzu Cypher query.
The query should search for relevant problems and concepts based on the user's question.
Return ONLY the Cypher query, no explanations.

User Question: {user_input}

Cypher Query:"""
    
    try:
        print(f"[DEBUG] Sending prompt to OpenAI for input: '{user_input}'")
        print(f"[DEBUG] Using model: gpt-4.1")
        
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a database query expert specializing in Kuzu Cypher queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        print(f"[DEBUG] OpenAI response received successfully")
        cypher_query = response.choices[0].message.content.strip()
        print(f"[DEBUG] Raw OpenAI response: '{cypher_query}'")
        
        # Remove any markdown formatting if present
        if cypher_query.startswith("```"):
            cypher_query = cypher_query.split("```")[1]
            if cypher_query.startswith("cypher"):
                cypher_query = cypher_query[6:]
            cypher_query = cypher_query.strip()
            print(f"[DEBUG] After markdown cleanup: '{cypher_query}'")
        
        print(f"[DEBUG] Final generated query: '{cypher_query}'")
        return cypher_query
        
    except Exception as e:
        print(f"[DEBUG] Error generating Cypher query: {e}")
        print(f"[DEBUG] Error type: {type(e).__name__}")
        # Fallback to simple query
        fallback_query = f'MATCH (p:Problem)-[:ABOUT]->(c:Concept) WHERE p.text CONTAINS "{user_input}" OR c.id CONTAINS "{user_input.lower()}" RETURN p.id, p.text, p.type LIMIT 10;'
        print(f"[DEBUG] Using fallback query: '{fallback_query}'")
        return fallback_query

def format_kuzu_results(user_question: str, kuzu_results: list):
    """
    Use OpenAI to format raw Kuzu results into proper Answer objects
    """
    if not kuzu_results:
        return [Answer(summarized_answer="No relevant problems found in the knowledge base", confidence_score=0.0)]
    
    # Prepare the results for OpenAI
    results_text = ""
    for i, result in enumerate(kuzu_results, 1):
        results_text += f"Result {i}: {result}\n"
    
    prompt = f"""You are an AI assistant helping format database query results into helpful answers.

User's Question: "{user_question}"

Raw Database Results:
{results_text}

Your task:
1. Analyze these results and determine how well they answer the user's question
2. Format each relevant result into a clear, helpful answer
3. Assign a confidence score (0.0-1.0) based on how well each result matches the question
4. Return ONLY a JSON array of answers in this exact format:

[
  {{
    "summarized_answer": "Clear, helpful summary of the problem and solution",
    "confidence_score": 0.85
  }}
]

Rules:
- Focus on the most relevant results
- Make answers clear and actionable
- Higher confidence for exact matches, lower for partial matches
- Maximum 5 answers
- If results don't match the question well, use lower confidence scores

JSON Response:"""

    try:
        print(f"[DEBUG] Formatting results with OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that formats database results into clear answers. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"[DEBUG] OpenAI formatting response: {result_text}")
        
        # Parse the JSON response
        try:
            answers_data = json.loads(result_text)
            answers = [Answer(
                summarized_answer=ans["summarized_answer"], 
                confidence_score=float(ans["confidence_score"])
            ) for ans in answers_data]
            print(f"[DEBUG] Successfully parsed {len(answers)} formatted answers")
            return answers
            
        except json.JSONDecodeError as e:
            print(f"[DEBUG] JSON parsing error: {e}")
            # Fallback to basic formatting
            return [Answer(
                summarized_answer=f"Found {len(kuzu_results)} relevant problems in the knowledge base",
                confidence_score=0.7
            )]
            
    except Exception as e:
        print(f"[DEBUG] Error formatting results: {e}")
        # Fallback formatting
        return [Answer(
            summarized_answer=f"Found {len(kuzu_results)} problems related to your question",
            confidence_score=0.6
        )]

# Request Models
class RegisterRequest(BaseModel):
    model_name: str
    model_parameters: Optional[str] = None

# StatusRequest no longer needed - using query parameter

class AskRequest(BaseModel):
    uuid: str
    question_summary: str
    conversation_history: str

class ContributeRequest(BaseModel):
    uuid: str
    question_summary: str
    answer_summary: str
    conversation_history: str

class KuzuRequest(BaseModel):
    query: str

# Response Models
class RegisterResponse(BaseModel):
    uuid: str

class StatusResponse(BaseModel):
    seed: int
    leech: int

class Answer(BaseModel):
    summarized_answer: str
    confidence_score: float

class AskResponse(BaseModel):
    answers: List[Answer]
    status: StatusResponse

class ContributeResponse(BaseModel):
    seed: int
    leech: int

class KuzuResponse(BaseModel):
    result: str
    success: bool
    error: Optional[str] = None

# Endpoints
@app.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Always generate a new UUID
        new_uuid = str(uuid.uuid4())
        
        # Store the registration in database
        cursor.execute("""
            INSERT INTO registrations (uuid, model_name, model_parameters, seed_count, leech_count)
            VALUES (?, ?, ?, 0, 0)
        """, (new_uuid, request.model_name, request.model_parameters))
        
        conn.commit()
        conn.close()
        
        return RegisterResponse(uuid=new_uuid)
    
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def status(uuid: str = Query(..., description="UUID for authentication")):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Look up UUID in database
        cursor.execute("SELECT seed_count, leech_count FROM registrations WHERE uuid = ?", (uuid,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            seed_count, leech_count = result
            return StatusResponse(seed=seed_count, leech=leech_count)
        else:
            # UUID not found
            raise HTTPException(status_code=404, detail="UUID not found. Please register first.")
    
    except HTTPException:
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Verify UUID exists and get current counts
        cursor.execute("SELECT seed_count, leech_count FROM registrations WHERE uuid = ?", (request.uuid,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise HTTPException(status_code=404, detail="UUID not found. Please register first.")
        
        seed_count, leech_count = result
        
        # Increment leech count (user is consuming knowledge)
        new_leech_count = leech_count + 1
        cursor.execute("UPDATE registrations SET leech_count = ? WHERE uuid = ?", (new_leech_count, request.uuid))
        conn.commit()
        
        # Execute Kuzu query to get real answers
        try:
            print(f"[DEBUG] Connecting to Kuzu database...")
            # Connect to the Kuzu database
            kuzu_db = kuzu.Database("/Users/barathwajanandan/Documents/OverFit/kuzu/overfit-db")
            kuzu_conn = kuzu.Connection(kuzu_db)
            print(f"[DEBUG] Connected to Kuzu database successfully")
            
            # Generate intelligent Cypher query using OpenAI
            print(f"[DEBUG] Generating Cypher query for: '{request.question_summary}'")
            kuzu_query = generate_cypher_query(request.question_summary)
            print(f"[DEBUG] Generated Cypher query: {kuzu_query}")
            
            # Execute the query
            print(f"[DEBUG] Executing Kuzu query...")
            kuzu_result = kuzu_conn.execute(kuzu_query)
            print(f"[DEBUG] Query executed successfully")
            
            # Collect all raw results from Kuzu
            raw_results = []
            row_count = 0
            while kuzu_result.has_next():
                row = kuzu_result.get_next()
                row_count += 1
                print(f"[DEBUG] Row {row_count}: {row}")
                raw_results.append(row)
            
            print(f"[DEBUG] Total rows collected: {row_count}")
            
            # Close Kuzu connection
            kuzu_conn.close()
            print(f"[DEBUG] Kuzu connection closed")
            
            # Use OpenAI to format the results into proper Answer objects
            print(f"[DEBUG] Using OpenAI to format {len(raw_results)} raw results...")
            answers = format_kuzu_results(request.question_summary, raw_results)
                
        except Exception as kuzu_error:
            print(f"[DEBUG] Kuzu error occurred: {kuzu_error}")
            print(f"[DEBUG] Error type: {type(kuzu_error).__name__}")
            # If Kuzu query fails, return error as answer
            answers = [Answer(summarized_answer=f"Error querying knowledge base: {str(kuzu_error)}", confidence_score=0.0)]
        
        conn.close()
        return AskResponse(
            answers=answers,
            status=StatusResponse(seed=seed_count, leech=new_leech_count)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/contribute", response_model=ContributeResponse)
async def contribute(request: ContributeRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Verify UUID exists and get current counts
        cursor.execute("SELECT seed_count, leech_count FROM registrations WHERE uuid = ?", (request.uuid,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise HTTPException(status_code=404, detail="UUID not found. Please register first.")
        
        seed_count, leech_count = result
        
        # Increment seed count (user is contributing knowledge)
        new_seed_count = seed_count + 1
        cursor.execute("UPDATE registrations SET seed_count = ? WHERE uuid = ?", (new_seed_count, request.uuid))
        conn.commit()
        
        conn.close()
        return ContributeResponse(seed=new_seed_count, leech=leech_count)
    
    except HTTPException:
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/kuzu", response_model=KuzuResponse)
async def execute_kuzu_query(request: KuzuRequest):
    try:
        # Connect to the Kuzu database
        db = kuzu.Database("/Users/barathwajanandan/Documents/OverFit/kuzu/overfit-db")
        conn = kuzu.Connection(db)
        
        # Execute the query
        result = conn.execute(request.query)
        
        # Convert the result to a string format
        result_str = ""
        while result.has_next():
            row = result.get_next()
            result_str += str(row) + "\n"
        
        # Close the connection
        conn.close()
        
        return KuzuResponse(
            result=result_str.strip() if result_str else "No results found",
            success=True,
            error=None
        )
    
    except Exception as e:
        return KuzuResponse(
            result="",
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3003)
