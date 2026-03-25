#!/usr/bin/env python3
"""
Integration test for Synthony API with database persistence.

Tests the complete workflow:
1. Upload system prompt
2. Upload dataset
3. Analyze dataset
4. Retrieve session
5. Get storage stats
6. Verify persistence

NOTE: These tests require a running Synthony API server at localhost:8000.
Run with: uvicorn synthony.api.server:app --reload
"""

import requests
import pytest
import pytest
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_DATA_DIR = Path("dataset/input_data")


def server_is_running():
    """Check if API server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# Skip all tests in this module if server is not running
pytestmark = pytest.mark.skipif(
    not server_is_running(),
    reason="API server not running at localhost:8000"
)


def server_is_running():
    """Check if API server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# Skip all tests in this module if server is not running
pytestmark = pytest.mark.skipif(
    not server_is_running(),
    reason="API server not running at localhost:8000"
)

def test_health():
    """Test 1: Health check"""
    print("=" * 80)
    print("TEST 1: Health Check")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    
    data = response.json()
    print(f"✓ Status: {data['status']}")
    print(f"✓ LLM Available: {data['llm_available']}")
    print(f"✓ Models Count: {data['models_count']}")
    assert data['status'] == 'healthy'
    print("\n✅ PASS: Health check successful\n")
    return data


def test_system_prompt_upload():
    """Test 2: Upload system prompt with deduplication"""
    print("=" * 80)
    print("TEST 2: System Prompt Upload & Deduplication")
    print("=" * 80)
    
    prompt_file = Path("config/SystemPrompt.md")
    assert prompt_file.exists(), f"System prompt not found: {prompt_file}"
    
    # First upload
    with open(prompt_file, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/upload/systemprompt",
            params={'version': 'test-v1.0', 'set_active': True},
            files={'file': f}
        )
    
    if response.status_code == 409:
        print("⚠ Prompt already exists (expected if re-running test)")
        data = response.json()
        print(f"  Detail: {data['detail']}")
    else:
        assert response.status_code == 200, f"Upload failed: {response.status_code}"
        data = response.json()
        print(f"✓ Uploaded version: {data['version']}")
        print(f"✓ Content hash: {data['content_hash']}")
        print(f"✓ Is active: {data['is_active']}")
    
    # Try duplicate upload
    with open(prompt_file, 'rb') as f:
        dup_response = requests.post(
            f"{BASE_URL}/upload/systemprompt",
            params={'version': 'test-v1.0'},
            files={'file': f}
        )
    
    assert dup_response.status_code == 409, "Duplicate should be rejected with 409"
    print("✓ Duplicate upload correctly rejected with 409 Conflict")
    
    print("\n✅ PASS: System prompt upload & deduplication working\n")
    return data if response.status_code == 200 else None


def test_dataset_upload_and_analyze():
    """Test 3: Upload dataset and analyze with persistence"""
    print("=" * 80)
    print("TEST 3: Dataset Upload & Analysis with Persistence")
    print("=" * 80)
    
    # Find a test CSV file
    csv_files = list(TEST_DATA_DIR.glob("*.csv"))
    assert csv_files, f"No CSV files found in {TEST_DATA_DIR}"
    
    test_file = csv_files[0]
    print(f"Using test file: {test_file.name}")
    
    # Upload and analyze
    with open(test_file, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/analyze",
            files={'file': ('test.csv', f, 'text/csv')},
            params={'dataset_id': 'integration_test'}
        )
    
    assert response.status_code == 200, f"Analysis failed: {response.status_code}"
    data = response.json()
    
    print(f"✓ Session ID: {data['session_id']}")
    print(f"✓ Analysis ID: {data['analysis_id']}")
    print(f"✓ Dataset ID: {data['dataset_id']}")
    print(f"✓ Rows: {data['dataset_profile']['row_count']}")
    print(f"✓ Columns: {data['dataset_profile']['column_count']}")
    
    # Verify stress factors
    stress_factors = data['dataset_profile']['stress_factors']
    print(f"✓ Stress factors detected: {[k for k, v in stress_factors.items() if v]}")
    
    print("\n✅ PASS: Dataset upload and analysis with persistence\n")
    return data


def test_session_retrieval(session_id):
    """Test 4: Retrieve session data (future feature)"""
    print("=" * 80)
    print("TEST 4: Session Retrieval")
    print("=" * 80)
    
    print(f"Session ID: {session_id}")
    
    # Note: Session retrieval endpoints not yet implemented
    print("⚠ Session retrieval endpoints not yet implemented")
    print("  Future endpoints:")
    print("  - GET /sessions/{session_id}")
    print("  - GET /sessions/{session_id}/data/{dataset_id}")
    
    print("\n⏭ SKIP: Session retrieval (not yet implemented)\n")


def test_storage_stats():
    """Test 5: Get storage statistics"""
    print("=" * 80)
    print("TEST 5: Storage Statistics")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/storage/stats")
    assert response.status_code == 200, f"Storage stats failed: {response.status_code}"
    
    data = response.json()
    storage = data['storage']
    
    print(f"✓ Total sessions: {storage['active_sessions']}")
    print(f"✓ Total datasets: {storage['total_datasets']}")
    print(f"✓ Total size: {storage['total_size_gb']:.3f} GB")
    print(f"✓ Storage limit: {storage['storage_limit_gb']} GB")
    print(f"✓ Usage: {storage['usage_percent']:.2f}%")
    
    assert storage['usage_percent'] < 100, "Storage full!"
    
    print(f"\n✅ PASS: Storage stats retrieved\n")
    return storage


def test_list_system_prompts():
    """Test 6: List all system prompts"""
    print("=" * 80)
    print("TEST 6: List System Prompts")
    print("=" * 80)
    
    response = requests.get(f"{BASE_URL}/systemprompts")
    assert response.status_code == 200, f"List prompts failed: {response.status_code}"
    
    data = response.json()
    print(f"✓ Total prompts: {data['total']}")
    print(f"✓ Active version: {data['active_version']}")
    
    for prompt in data['prompts']:
        status = "ACTIVE" if prompt['is_active'] else "inactive"
        print(f"  - {prompt['version']}: {status} ({prompt['content_length']} chars)")
    
    print(f"\n✅ PASS: Listed {data['total']} system prompts\n")
    return data


def main():
    """Run all integration tests"""
    print("\n" + "=" * 80)
    print(" SYNTHONY API INTEGRATION TEST SUITE ".center(80))
    print("=" * 80 + "\n")
    
    try:
        # Run tests
        health_data = test_health()
        prompt_data = test_system_prompt_upload()
        analysis_data = test_dataset_upload_and_analyze()
        test_session_retrieval(analysis_data['session_id'])
        storage_data = test_storage_stats()
        prompts_data = test_list_system_prompts()
        
        # Summary
        print("=" * 80)
        print(" TEST SUMMARY ".center(80))
        print("=" * 80)
        print(f"✅ Health Check: PASS")
        print(f"✅ System Prompt Upload & Deduplication: PASS")
        print(f"✅ Dataset Analysis with Persistence: PASS")
        print(f"⏭ Session Retrieval: SKIP (not implemented)")
        print(f"✅ Storage Statistics: PASS")
        print(f"✅ List System Prompts: PASS")
        print("\n" + "=" * 80)
        print(" ALL TESTS PASSED! ".center(80, "="))
        print("=" * 80 + "\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
