#!/usr/bin/env python3
"""
Simple test script to verify the backend API is working
"""

import requests
import sys

def test_api():
    """Test the backend API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Document Visual Parser API")
    print("=" * 50)
    
    # Test health check
    try:
        print("1. Testing health check endpoint...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the backend server is running: python run_backend.py")
        return False
    
    # Test API documentation
    try:
        print("\n2. Testing API documentation...")
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("✅ API documentation is available")
            print(f"   Visit: {base_url}/docs")
        else:
            print(f"⚠️  API documentation returned: {response.status_code}")
    except Exception as e:
        print(f"⚠️  API documentation test failed: {e}")
    
    print("\n✅ Basic API tests completed successfully!")
    print(f"🌐 API is running at: {base_url}")
    print(f"📖 Documentation: {base_url}/docs")
    
    return True

if __name__ == "__main__":
    success = test_api()
    sys.exit(0 if success else 1) 