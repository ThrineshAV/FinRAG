#!/usr/bin/env python3
"""Simple test to verify everything works."""

import subprocess
import time
import requests
import sys

def main():
    print("Starting server...")

    # Start server
    server = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "src.api:app",
        "--host", "127.0.0.1", "--port", "8000"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Wait for server to start
        time.sleep(3)

        # Test health
        print("Testing /health...")
        resp = requests.get("http://127.0.0.1:8000/health")
        print(f"Health: {resp.status_code} - {resp.json()}")

        # Test admin keys with test123
        print("\nTesting /admin/keys with test123...")
        resp = requests.get(
            "http://127.0.0.1:8000/admin/keys",
            headers={"X-API-Key": "test123"}
        )
        print(f"Admin keys: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Response: {resp.json()}")
        else:
            print(f"Error: {resp.text}")

        # Test create key
        print("\nTesting POST /admin/keys...")
        resp = requests.post(
            "http://127.0.0.1:8000/admin/keys",
            headers={
                "X-API-Key": "test123",
                "Content-Type": "application/json"
            },
            json={"name": "test-key", "role": "reader"}
        )
        print(f"Create key: {resp.status_code}")
        if resp.status_code == 200:
            print(f"Response: {resp.json()}")
        else:
            print(f"Error: {resp.text}")

    finally:
        # Stop server
        server.terminate()
        server.wait()
        print("\nServer stopped.")

if __name__ == "__main__":
    main()