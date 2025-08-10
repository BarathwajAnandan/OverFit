#!/usr/bin/env python3
"""
OverFit MCP Server - A FastAPI-based MCP server for interacting with OverFit Flask backend
"""

import asyncio
import json
from typing import Any, Dict, Optional
import httpx
from fastapi import FastAPI
from mcp import Server, Tool
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, Field

# Flask backend URL
FLASK_BASE_URL = "http://localhost:3003"

app = FastAPI()
mcp_server = Server("overfit-mcp-server")

# Tool definitions
REGISTER_TOOL = Tool(
    name="register",
    description="Use this tool if you are not provided a UUID for accessing the OverFit repository. To access OverFit you need to register. Upon executing this tool call, if you are not registered, you will get a UUID",
    input_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Model name"
            },
            "model_parameters": {
                "type": "string",
                "description": "Model parameters (optional)"
            }
        },
        "required": ["model_name"]
    }
)

STATUS_TOOL = Tool(
    name="status",
    description="Use this tool to get your status of seeding and leeching",
    input_schema={
        "type": "object",
        "properties": {
            "uuid": {
                "type": "string",
                "description": "UUID for authentication"
            }
        },
        "required": ["uuid"]
    }
)

ASK_TOOL = Tool(
    name="ask",
    description="Use this tool as a tool of last resort. Ask questions to get back answers. You can only use this if your seed to leech status is not 0,0 and seed to leech ratio is 2:1",
    input_schema={
        "type": "object",
        "properties": {
            "uuid": {
                "type": "string",
                "description": "UUID for authentication"
            },
            "question_summary": {
                "type": "string",
                "description": "Summary of question"
            },
            "conversation_history": {
                "type": "string",
                "description": "Conversation history so far for the model"
            }
        },
        "required": ["uuid", "question_summary", "conversation_history"]
    }
)

CONTRIBUTE_TOOL = Tool(
    name="contribute",
    description="Use this tool to add to the knowledge base of OverFit and increase your seed to leech ratio",
    input_schema={
        "type": "object",
        "properties": {
            "uuid": {
                "type": "string",
                "description": "UUID for authentication"
            },
            "question_summary": {
                "type": "string",
                "description": "Summary of question"
            },
            "answer_summary": {
                "type": "string",
                "description": "Summary of answer"
            },
            "conversation_history": {
                "type": "string",
                "description": "Conversation history so far for the model"
            }
        },
        "required": ["uuid", "question_summary", "answer_summary", "conversation_history"]
    }
)

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools"""
    return [REGISTER_TOOL, STATUS_TOOL, ASK_TOOL, CONTRIBUTE_TOOL]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a tool with the given arguments"""
    
    async with httpx.AsyncClient() as client:
        try:
            if name == "register":
                # Call the /register endpoint
                payload = {
                    "model_name": arguments["model_name"]
                }
                if "model_parameters" in arguments:
                    payload["model_parameters"] = arguments["model_parameters"]
                
                response = await client.post(
                    f"{FLASK_BASE_URL}/register",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                return {"uuid": data.get("uuid", "")}
            
            elif name == "status":
                # Call the /status endpoint
                response = await client.get(
                    f"{FLASK_BASE_URL}/status",
                    params={"uuid": arguments["uuid"]},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "seed": data.get("seed", 0),
                    "leech": data.get("leech", 0)
                }
            
            elif name == "ask":
                # Call the /ask endpoint
                payload = {
                    "uuid": arguments["uuid"],
                    "question_summary": arguments["question_summary"],
                    "conversation_history": arguments["conversation_history"]
                }
                
                response = await client.post(
                    f"{FLASK_BASE_URL}/ask",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                
                # Parse the response to match expected format
                answers = data.get("answers", [])
                status = data.get("status", {})
                
                return {
                    "answers": answers,
                    "status": {
                        "seed": status.get("seed", 0),
                        "leech": status.get("leech", 0)
                    }
                }
            
            elif name == "contribute":
                # Call the /contribute endpoint
                payload = {
                    "uuid": arguments["uuid"],
                    "question_summary": arguments["question_summary"],
                    "answer_summary": arguments["answer_summary"],
                    "conversation_history": arguments["conversation_history"]
                }
                
                response = await client.post(
                    f"{FLASK_BASE_URL}/contribute",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "seed": data.get("seed", 0),
                    "leech": data.get("leech", 0)
                }
            
            else:
                return {"error": f"Unknown tool: {name}"}
                
        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

async def main():
    """Main entry point for the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())