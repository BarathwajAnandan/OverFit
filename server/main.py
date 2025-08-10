from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import uvicorn

app = FastAPI(title="Agent Marketplace API")

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
    return RegisterResponse(uuid=str(uuid.uuid4()))

@app.get("/status", response_model=StatusResponse)
async def status(uuid: str = Query(..., description="UUID for authentication")):
    return StatusResponse(seed=42, leech=15)

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    mock_answers = [
        Answer(summarized_answer="This is a mock answer to your question", confidence_score=0.85),
        Answer(summarized_answer="Here's another perspective on the issue", confidence_score=0.72),
        Answer(summarized_answer="not found", confidence_score=0.0)
    ]
    
    return AskResponse(
        answers=mock_answers,
        status=StatusResponse(seed=42, leech=16)
    )

@app.post("/contribute", response_model=ContributeResponse)
async def contribute(request: ContributeRequest):
    return ContributeResponse(seed=43, leech=15)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=3003)
