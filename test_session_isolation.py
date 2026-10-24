import os
os.environ.setdefault("FLASK_ENV", "testing")


import app as app_module
from app import app, _session_contexts, _session_context_lock

app.config["TESTING"] = True


def _csrf_token(client):
    response = client.get("/csrf-token")
    assert response.status_code == 200
    return response.get_json()["csrf_token"]



def _session_id_from_client(client):
    response = client.get("/session-id")
    assert response.status_code == 200
    return response.get_json()["session_id"]


def test_two_clients_receive_different_session_ids():
    with app.test_client() as c1:
        c1.get("/")
        sid1 = _session_id_from_client(c1)
    with app.test_client() as c2:
        c2.get("/")
        sid2 = _session_id_from_client(c2)
    assert sid1 != sid2


def test_conversation_context_is_isolated():
    with app.test_client() as c1:
        c1.post("/ask", data={"message": "I feel sad and overwhelmed"}, headers={"X-CSRFToken": _csrf_token(c1)})
        sid1 = _session_id_from_client(c1)
    with app.test_client() as c2:
        c2.post("/ask", data={"message": "Hello there"}, headers={"X-CSRFToken": _csrf_token(c2)})
        sid2 = _session_id_from_client(c2)

    with _session_context_lock:
        ctx1 = _session_contexts[sid1]
        ctx2 = _session_contexts[sid2]

    assert ctx1["mood"] == "negative"
    assert ctx2["mood"] != "negative"


def test_feedback_cross_session_is_rejected():
    with app.test_client() as c1:
        c1.post("/ask", data={"message": "I feel anxious"}, headers={"X-CSRFToken": _csrf_token(c1)})
        sid1 = _session_id_from_client(c1)

    with app.test_client() as c2:
        response = c2.post(
            "/feedback",
            json={
                "response_id": "test-response",
                "session_id": sid1,
                "feedback": 1,
            },
        )
        assert response.status_code == 403


def test_clear_feedback_cross_session_is_rejected():
    with app.test_client() as c1:
        c1.post("/ask", data={"message": "I feel anxious"}, headers={"X-CSRFToken": _csrf_token(c1)})
        sid1 = _session_id_from_client(c1)

    with app.test_client() as c2:
        response = c2.post(
            "/clear-feedback",
            json={"session_id": sid1},
        )
        assert response.status_code == 403


def test_session_cookie_configuration():
    assert app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"