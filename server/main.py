from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import uvicorn
import sqlite3
from datetime import datetime

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
        
        # Mock answers for now
        mock_answers = [
            Answer(summarized_answer="This is a mock answer to your question", confidence_score=0.85),
            Answer(summarized_answer="Here's another perspective on the issue", confidence_score=0.72),
            Answer(summarized_answer="not found", confidence_score=0.0)
        ]
        
        conn.close()
        return AskResponse(
            answers=mock_answers,
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

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3003)
