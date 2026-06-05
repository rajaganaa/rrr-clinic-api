"""Basic smoke tests for RRR Clinic API."""
import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_health_endpoint():
    """Health check must return 200 and correct service name."""
    from main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "RRR Clinic" in data["service"]

def test_root_endpoint():
    from main import app
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
