#!/usr/bin/env python3
"""Test script to verify admin endpoints are working."""

import sys
import requests
import time

def test_server():
    base_url = "http://127.0.0.1:8000"

    print("=" * 60)
    print("FINSIGHT-RAG API VERIFICATION")
    print("=" * 60)

    # Test health endpoint
    print("\n1. Testing /health endpoint...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"   ✅ Status: {resp.status_code}")
        print(f"   ✅ Response: {resp.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Test admin keys with correct key
    print("\n2. Testing /admin/keys with correct key...")
    try:
        resp = requests.get(
            f"{base_url}/admin/keys",
            headers={"X-API-Key": "test123"},
            timeout=5
        )
        print(f"   ✅ Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   ✅ Response: {resp.json()}")
        else:
            print(f"   ❌ Error: {resp.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Test create key endpoint
    print("\n3. Testing POST /admin/keys (create key)...")
    try:
        resp = requests.post(
            f"{base_url}/admin/keys",
            headers={
                "X-API-Key": "test123",
                "Content-Type": "application/json"
            },
            json={"name": "test-new-key", "role": "reader"},
            timeout=5
        )
        print(f"   ✅ Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   ✅ Response: {resp.json()}")
        else:
            print(f"   ❌ Error: {resp.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Test delete key endpoint
    print("\n4. Testing DELETE /admin/keys/{key_id}...")
    try:
        resp = requests.delete(
            f"{base_url}/admin/keys/admin001",
            headers={"X-API-Key": "test123"},
            timeout=5
        )
        print(f"   ✅ Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   ✅ Response: {resp.json()}")
        else:
            print(f"   ❌ Error: {resp.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    # Wait a bit for server to start if needed
    time.sleep(2)
    success = test_server()
    sys.exit(0 if success else 1)