"""
Quick smoke test — run after starting the server to verify it works end-to-end.
Usage:  python test_api.py
"""
import requests

BASE = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE}/health")
    print("HEALTH:", r.json())
    assert r.status_code == 200


def test_ingest_text():
    # Create a tiny test text file
    with open("test_doc.txt", "w") as f:
        f.write(
            "VVDN Technologies is a global product engineering company. "
            "The KMS portal stores internal documentation, meeting recordings, "
            "and design diagrams. Project Atlas was approved on March 5, 2026 "
            "with a budget of 2 crore INR."
        )
    with open("test_doc.txt", "rb") as f:
        r = requests.post(f"{BASE}/ingest", files={"file": f})
    print("INGEST:", r.json())
    assert r.status_code == 200


def test_chat():
    r = requests.post(
        f"{BASE}/chat",
        data={"query": "When was Project Atlas approved and what's the budget?"},
    )
    print("CHAT:", r.json())
    assert r.status_code == 200


def test_stats():
    r = requests.get(f"{BASE}/kb/stats")
    print("KB STATS:", r.json())


if __name__ == "__main__":
    test_health()
    test_ingest_text()
    test_chat()
    test_stats()
    print("\n✅ All tests passed.")
