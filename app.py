import hmac
import os
import secrets
import sys
import re
import sqlite3
import threading
import time
import uuid
from datetime import timedelta

from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify, session

import json
import random

import torch

from model import NeuralNet
from nlp_utils import tokenize, bag_of_words
from safety import get_crisis_response, is_safe_to_proceed, resolve_crisis_region
from database import (
    create_tables,
    save_interaction,
    generate_session_id,
    register_response,
    get_response_owner,
    record_feedback,
)


app = Flask(__name__)

# Require a real secret key in production. A development-only fallback is
# intentionally weak and must never be used for a deployed instance.
def _resolve_secret_key():
    secret = os.environ.get("FLASK_SECRET_KEY")
    if secret:
        return secret

    env = os.environ.get("FLASK_ENV", "").lower()

    # Production must never use a predictable fallback secret.
    if env == "production":
        raise RuntimeError(
            "FLASK_SECRET_KEY must be set for production deployments"
        )

    # Explicit test configuration may use the documented non-production
    # fallback. This is never enabled by pytest/process detection alone.
    explicit_test = env in {"testing", "test"} or os.environ.get(
        "MINDMATE_TEST_MODE", ""
    ).lower() in {"1", "true", "yes"} or os.environ.get(
        "MINDMATE_ALLOW_TEST_SECRET", ""
    ).lower() in {"1", "true", "yes"}

    if explicit_test:
        return "mindmate-development-only-secret-not-for-production"

    # Explicit local-development opt-in.
    if env == "development" and os.environ.get(
        "MINDMATE_ALLOW_DEV_SECRET", ""
    ).lower() in {"1", "true", "yes"}:
        return "mindmate-development-only-secret-not-for-production"

    raise RuntimeError(
        "FLASK_SECRET_KEY must be set for non-development deployments"
    )


_secret_key = _resolve_secret_key()
app.secret_key = _secret_key

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_SECURE_COOKIES", "").lower()
    in {"1", "true", "yes"}
    or os.environ.get("FLASK_ENV", "").lower() == "production",
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)

# --------------------------------------------------
# Configuration and model loading
# --------------------------------------------------

with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

db_path = "data/chatbot.db"

# Initialize database tables
create_tables(db_path)

# Load the existing trained model.
# Do not retrain or overwrite the checkpoint here.
data = torch.load("data/data.pth", map_location=torch.device("cpu"))

model = NeuralNet(
    data["input_size"],
    data["hidden_size"],
    data["output_size"],
)

model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags = data["tags"]

# --------------------------------------------------
# Conversation state
# --------------------------------------------------

_session_contexts = {}
_session_context_lock = threading.Lock()
_MAX_SESSION_CONTEXTS = 1000


def _uuid4_pattern(value):
    return isinstance(value, str) and bool(
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            value,
        )
    )


def get_session_id():
    """Return the signed, server-issued session ID for this request."""
    sid = session.get("session_id")
    if not _uuid4_pattern(sid):
        sid = str(uuid.uuid4())
        session["session_id"] = sid
        session.permanent = True
    return sid


def _cleanup_session_contexts():
    if len(_session_contexts) <= _MAX_SESSION_CONTEXTS:
        return
    # Remove the least-recently-used entries when the store grows too large.
    ordered = sorted(
        _session_contexts.items(),
        key=lambda item: item[1].get("last_used", 0.0),
    )
    for sid, _ in ordered[: len(_session_contexts) - _MAX_SESSION_CONTEXTS]:
        _session_contexts.pop(sid, None)


def get_conversation_context(session_id):
    """Return a snapshot of the conversation context for one session."""
    with _session_context_lock:
        context = _session_contexts.setdefault(
            session_id,
            {
                "mood": "neutral",
                "recent_tags": [],
                "topics": set(),
                "last_used": 0.0,
            },
        )
        context["last_used"] = time.time()
        return {
            "mood": context["mood"],
            "recent_tags": list(context["recent_tags"][-3:]),
            "topics": list(context["topics"])[:3],
        }


def update_conversation_context(session_id, tag, user_input):
    """Update conversation mood, recent intents, and topics for one session."""
    with _session_context_lock:
        context = _session_contexts.setdefault(
            session_id,
            {
                "mood": "neutral",
                "recent_tags": [],
                "topics": set(),
                "last_used": 0.0,
            },
        )

        if tag in ["mental_health", "coping_strategies", "sad"]:
            context["mood"] = "negative"
        elif tag in ["greeting", "thanks", "positive"]:
            if context["mood"] == "neutral":
                context["mood"] = "positive"

        context["recent_tags"].append(tag)
        context["recent_tags"] = context["recent_tags"][-10:]

        if len(user_input) > 10:
            words = user_input.lower().split()
            context["topics"].update(words[:5])

            if len(context["topics"]) > 100:
                context["topics"] = set(list(context["topics"])[-50:])

        context["last_used"] = time.time()
        _cleanup_session_contexts()


# --------------------------------------------------
# Intent prediction
# --------------------------------------------------

def predict_intent(sentence):
    """Predict the user's intent and return its confidence."""
    tokens = tokenize(sentence)
    bow = bag_of_words(tokens, all_words)

    bow = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(bow)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

    return tags[prediction.item()], confidence.item()


# --------------------------------------------------
# Context-aware responses
# --------------------------------------------------

def get_response_with_context(tag, conf, context):
    """Generate an intent response with basic conversation context."""

    for intent in intents["intents"]:
        if tag == intent["tag"]:
            responses = intent.get("responses", [])
            if not responses:
                return (
                    "I'm here to listen and support you. "
                    "Could you tell me a little more about what's going on?"
                )

            base_response = random.choice(responses)

            # Add a supportive acknowledgement when appropriate.
            if context.get("mood") == "negative" and tag != "mental_health":
                supportive_addons = [
                    " I'm sorry you're going through this.",
                    " That sounds really challenging.",
                    " Thank you for sharing that with me.",
                ]
                base_response += random.choice(supportive_addons)

            recent_tags_list = context["recent_tags"]
            topics = context["topics"]

            # Follow up on previously discussed coping topics.
            if (
                "coping" in str(topics).lower()
                and tag != "coping_strategies"
            ):
                follow_ups = [
                    " How has that been going since we last talked about it?",
                    " Have you tried any of those strategies since?",
                ]
                base_response += random.choice(follow_ups)

            # Continue a previous mental-health conversation.
            if tag == "mental_health" and recent_tags_list:
                follow_ups = [
                    " How have you been feeling since we last talked?",
                    " I'm here if you want to continue that conversation.",
                ]
                base_response += random.choice(follow_ups)

            return base_response

    fallback_responses = [
        "I'm not sure I understand completely. Could you rephrase that?",
        "That's an interesting perspective. Tell me more about that.",
        "I want to make sure I understand you correctly.",
        "Let me reflect back what I'm hearing.",
    ]

    return random.choice(fallback_responses)


# --------------------------------------------------
# Database logging
# --------------------------------------------------

def log_to_db(user_input, predicted_tag, confidence, session_id=None):
    """Save an interaction using the current application session."""
    if session_id is None:
        session_id = get_session_id()

    conn = sqlite3.connect(db_path)

    try:
        save_interaction(
            conn,
            session_id,
            user_input,
            predicted_tag,
            confidence,
        )
    finally:
        conn.close()


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.before_request
def _csrf_origin_check():
    """Defense-in-depth against cross-site POSTs to state-changing endpoints."""
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None
    origin = request.headers.get("Origin")
    if origin:
        if urlparse(origin).netloc != request.host:
            return jsonify({"error": "Cross-site request blocked"}), 403

    # Require a session-bound CSRF token for cookie-authenticated writes.
    expected = session.get("csrf_token")
    provided = request.headers.get("X-CSRFToken")
    if not provided and request.form:
        provided = request.form.get("csrf_token")
    if not expected or not provided or not hmac.compare_digest(
        str(expected), str(provided)
    ):
        return jsonify({"error": "Missing or invalid CSRF token"}), 403
    return None


def _persist_response_ownership(response_id, sid):
    conn = sqlite3.connect(db_path)
    try:
        register_response(conn, response_id, sid)
    finally:
        conn.close()


@app.route("/csrf-token")
def csrf_token_endpoint():
    """Issue or return the CSRF token bound to the signed session."""
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
        session.permanent = True
    return jsonify({"csrf_token": token})


@app.route("/ask", methods=["POST"])
def ask():
    sid = get_session_id()
    user_msg = request.form.get("message", "")

    # Input validation: empty or whitespace-only message.
    if not user_msg or not user_msg.strip():
        return jsonify({
            "response": "Please enter a message to continue."
        }), 400

    user_msg = user_msg.strip()

    # Maximum message length.
    if len(user_msg) > 1000:
        return jsonify({
            "response": (
                "Your message is too long. "
                "Please keep it under 1000 characters."
            )
        }), 400

    # Crisis safety check must happen before ordinary intent responses.
    if not is_safe_to_proceed(user_msg):
        response = get_crisis_response(
            "immediate_danger", region=resolve_crisis_region()
        )

        response_id = str(uuid.uuid4())
        log_to_db(user_msg, "crisis", 1.0, sid)
        update_conversation_context(sid, "crisis", user_msg)
        _persist_response_ownership(response_id, sid)

        return jsonify({"response": response, "response_id": response_id})

    # Predict the ordinary intent only after the safety check.
    tag, confidence = predict_intent(user_msg)

    context = get_conversation_context(sid)

    # Update the context for subsequent messages.
    update_conversation_context(sid, tag, user_msg)

    if confidence > 0.4:
        response = get_response_with_context(tag, confidence, context)
    else:
        tag = "unknown"
        response = (
            "I'm not entirely sure I understand. I'd love to hear more "
            "about that. Tell me more, and I'll do my best to respond "
            "thoughtfully."
        )

    log_to_db(user_msg, tag, confidence, sid)

    response_id = str(uuid.uuid4())
    _persist_response_ownership(response_id, sid)

    return jsonify({"response": response, "response_id": response_id})


# --------------------------------------------------
# Feedback endpoint
# --------------------------------------------------

@app.route("/feedback", methods=["POST"])
def feedback():
    """Record feedback on a chatbot response."""
    sid = get_session_id()
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    response_id = data.get("response_id")
    requested_session_id = data.get("session_id")
    feedback_value = data.get("feedback")

    if requested_session_id is not None and requested_session_id != sid:
        return jsonify({"error": "Session identifier does not match this client"}), 403

    feedback_session_id = sid

    if not response_id:
        return jsonify({
            "error": "Missing required field: response_id"
        }), 400

    # Validate server-side ownership and record feedback in one connection/
    # transaction context.
    conn = sqlite3.connect(db_path)

    try:
        owner = get_response_owner(conn, response_id)
        if owner is None or owner != sid:
            return jsonify({"error": "Invalid or unauthorized response identifier"}), 403

        # Reject booleans and values other than integer 1 or -1.
        if (
            isinstance(feedback_value, bool)
            or not isinstance(feedback_value, int)
            or feedback_value not in (1, -1)
        ):
            return jsonify({
                "error": "Invalid feedback value. Must be 1 (helpful) or -1 (not helpful)"
            }), 400

        success = record_feedback(
            conn, response_id, feedback_session_id, feedback_value
        )
    except sqlite3.Error:
        conn.rollback()
        return jsonify({"error": "Database error while recording feedback"}), 500
    finally:
        conn.close()

    if not success:
        return jsonify({
            "error": "You have already submitted feedback for this response"
        }), 409

    return jsonify({
        "success": True,
        "message": "Feedback recorded successfully",
        "feedback": (
            "helpful" if feedback_value == 1 else "not helpful"
        ),
    })


# --------------------------------------------------
# Clear feedback endpoint
# --------------------------------------------------

@app.route("/clear-feedback", methods=["POST"])
def clear_feedback():
    """Clear feedback associated with a session."""
    sid = get_session_id()
    from database import clear_user_feedback

    data = request.get_json(silent=True) or {}
    requested_session_id = data.get("session_id", sid)

    if requested_session_id != sid:
        return jsonify({"error": "Session identifier does not match this client"}), 403

    if not isinstance(requested_session_id, str) or not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        requested_session_id,
    ):
        return jsonify({"error": "Invalid session identifier"}), 400

    conn = sqlite3.connect(db_path)

    try:
        clear_user_feedback(conn, requested_session_id)
    finally:
        conn.close()

    return jsonify({
        "success": True,
        "message": "Feedback cleared successfully",
    })


# --------------------------------------------------
# Application entry point
# --------------------------------------------------

@app.route("/session-id")
def session_id_endpoint():
    """Test-only helper for exposing the signed session ID."""
    if not (app.config.get("TESTING") or app.debug):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"session_id": get_session_id()})

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"})
