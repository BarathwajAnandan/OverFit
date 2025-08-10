#!/usr/bin/env python3
"""
Test client for the OverFit MCP server
Tests the MCP server by making direct HTTP requests to the Flask backend
"""

import asyncio
import httpx
import json
from typing import Optional

FLASK_BASE_URL = "http://localhost:3003"

async def test_register(client: httpx.AsyncClient) -> Optional[str]:
    """Test the register endpoint"""
    print("\n=== Testing Register ===")
    try:
        payload = {
            "model_name": "test-model",
            "model_parameters": "temperature=0.7"
        }
        response = await client.post(
            f"{FLASK_BASE_URL}/register",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        uuid = data.get("uuid")
        print(f"✓ Registration successful. UUID: {uuid}")
        return uuid
    except Exception as e:
        print(f"✗ Registration failed: {e}")
        return None

async def test_status(client: httpx.AsyncClient, uuid: str):
    """Test the status endpoint"""
    print("\n=== Testing Status ===")
    try:
        response = await client.get(
            f"{FLASK_BASE_URL}/status",
            params={"uuid": uuid},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        print(f"✓ Status retrieved: Seed={data.get('seed', 0)}, Leech={data.get('leech', 0)}")
        return data
    except Exception as e:
        print(f"✗ Status check failed: {e}")
        return None

async def test_contribute(client: httpx.AsyncClient, uuid: str):
    """Test the contribute endpoint"""
    print("\n=== Testing Contribute ===")
    try:
        payload = {
            "uuid": uuid,
            "question_summary": "What is Python decorators?",
            "answer_summary": "Python decorators are a way to modify or enhance functions without changing their code.",
            "conversation_history": "User asked about Python decorators, explained with examples."
        }
        response = await client.post(
            f"{FLASK_BASE_URL}/contribute",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        print(f"✓ Contribution successful: Seed={data.get('seed', 0)}, Leech={data.get('leech', 0)}")
        return data
    except Exception as e:
        print(f"✗ Contribution failed: {e}")
        return None

async def test_ask(client: httpx.AsyncClient, uuid: str):
    """Test the ask endpoint"""
    print("\n=== Testing Ask ===")
    try:
        payload = {
            "uuid": uuid,
            "question_summary": "How do Python decorators work?",
            "conversation_history": "User is learning about Python advanced features."
        }
        response = await client.post(
            f"{FLASK_BASE_URL}/ask",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        answers = data.get("answers", [])
        status = data.get("status", {})
        
        print(f"✓ Ask successful:")
        if answers:
            for i, answer in enumerate(answers, 1):
                print(f"  Answer {i}: {answer.get('summarized_answer', 'N/A')} (confidence: {answer.get('confidence_score', 0)})")
        else:
            print("  No answers returned")
        print(f"  Status: Seed={status.get('seed', 0)}, Leech={status.get('leech', 0)}")
        return data
    except Exception as e:
        print(f"✗ Ask failed: {e}")
        return None

async def main():
    """Run all tests"""
    print("Starting OverFit MCP Client Tests")
    print("=" * 50)
    print(f"Testing against Flask backend at: {FLASK_BASE_URL}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test server connectivity
        print("\n=== Checking Flask Backend ===")
        try:
            # Try a simple request to see if server is up
            await client.get(FLASK_BASE_URL, follow_redirects=False)
            print("✓ Flask backend is reachable")
        except Exception as e:
            print(f"✗ Cannot reach Flask backend at {FLASK_BASE_URL}")
            print(f"  Error: {e}")
            print("\nMake sure your Flask server is running on port 3003")
            return
        
        # Run tests
        uuid = await test_register(client)
        
        if uuid:
            await test_status(client, uuid)
            
            # Contribute a few times to build up seed ratio
            print("\n=== Building Seed Ratio ===")
            for i in range(2):
                await test_contribute(client, uuid)
            
            # Check status after contributions
            await test_status(client, uuid)
            
            # Try asking a question
            await test_ask(client, uuid)
        else:
            print("\n✗ Cannot proceed without UUID from registration")
    
    print("\n" + "=" * 50)
    print("Tests completed!")

if __name__ == "__main__":
    asyncio.run(main())