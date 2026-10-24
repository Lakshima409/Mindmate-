import os
os.environ.setdefault("FLASK_ENV", "testing")


"""Regression tests for empty intent response lists."""
import app as app_module
from app import get_response_with_context

EMPTY_RESPONSE_TAGS = [
    "bipolar",
    "anxiety",
    "personality_disorder",
    "stress",
    "suicidal",
    "depression",
    "normal",
]

CONTEXT = {"mood": "neutral", "recent_tags": [], "topics": []}


def test_empty_response_intents_return_fallback():
    for tag in EMPTY_RESPONSE_TAGS:
        response = get_response_with_context(tag, 0.9, CONTEXT)
        assert isinstance(response, str)
        assert response.strip()
        assert "I'm here to listen" in response


def test_empty_response_route_does_not_return_500():
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    token_response = client.get("/csrf-token")
    token = token_response.get_json()["csrf_token"]
    response = client.post(
        "/ask",
        data={"message": "I am feeling stressed about exams"},
        headers={"X-CSRFToken": token},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert "response" in payload
    assert payload["response"].strip()