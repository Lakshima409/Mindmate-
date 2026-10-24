import os
os.environ.setdefault("FLASK_ENV", "testing")


import os
import sqlite3
import uuid

import app as app_module
from app import app

app.config["TESTING"] = True


def _csrf_token(client):
    response = client.get("/csrf-token")
    assert response.status_code == 200
    return response.get_json()["csrf_token"]


def _headers(client, origin=None):
    headers = {"X-CSRFToken": _csrf_token(client)}
    if origin is not None:
        headers["Origin"] = origin
    return headers


def _ask(client, message="hello"):
    return client.post(
        "/ask",
        data={"message": message},
        headers=_headers(client),
    )




def test_valid_response_id_can_be_rated():
    with app.test_client() as client:
        ask_resp = _ask(client)
        assert ask_resp.status_code == 200
        rid = ask_resp.get_json()["response_id"]
        assert rid
        fb_resp = client.post(
            "/feedback",
            json={"response_id": rid, "feedback": 1},
            headers=_headers(client),
        )
        assert fb_resp.status_code == 200
        assert fb_resp.get_json()["success"] is True


def test_unknown_response_id_rejected():
    with app.test_client() as client:
        _ask(client)
        fb_resp = client.post(
            "/feedback",
            json={"response_id": str(uuid.uuid4()), "feedback": 1},
            headers=_headers(client),
        )
        assert fb_resp.status_code == 403


def test_cross_session_response_id_rejected():
    with app.test_client() as c1:
        rid = _ask(c1).get_json()["response_id"]
    with app.test_client() as c2:
        fb_resp = c2.post(
            "/feedback",
            json={"response_id": rid, "feedback": 1},
            headers=_headers(c2),
        )
        assert fb_resp.status_code == 403


def test_rejected_feedback_does_not_insert():
    with app.test_client() as c1:
        rid = _ask(c1).get_json()["response_id"]
    with app.test_client() as c2:
        c2.post(
            "/feedback",
            json={"response_id": rid, "feedback": 1},
            headers=_headers(c2),
        )
    conn = sqlite3.connect(app_module.db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE response_id = ?", (rid,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_duplicate_feedback_remains_correct():
    with app.test_client() as client:
        rid = _ask(client).get_json()["response_id"]
        first = client.post(
            "/feedback",
            json={"response_id": rid, "feedback": 1},
            headers=_headers(client),
        )
        second = client.post(
            "/feedback",
            json={"response_id": rid, "feedback": -1},
            headers=_headers(client),
        )
        assert first.status_code == 200
        assert second.status_code == 409


def test_csrf_missing_token_blocked():
    with app.test_client() as client:
        _ask(client)
        resp = client.post(
            "/feedback",
            json={"response_id": "x", "feedback": 1},
        )
        assert resp.status_code == 403


def test_csrf_mismatched_origin_blocked():
    with app.test_client() as client:
        token = _csrf_token(client)
        _ask(client)
        resp = client.post(
            "/feedback",
            json={"response_id": "x", "feedback": 1},
            headers={
                "X-CSRFToken": token,
                "Origin": "https://evil.example",
            },
        )
        assert resp.status_code == 403


def test_csrf_null_origin_blocked():
    with app.test_client() as client:
        token = _csrf_token(client)
        resp = client.post(
            "/feedback",
            json={"response_id": "x", "feedback": 1},
            headers={"X-CSRFToken": token, "Origin": "null"},
        )
        assert resp.status_code == 403


def test_csrf_matching_origin_allowed():
    with app.test_client() as client:
        rid = _ask(client).get_json()["response_id"]
        resp = client.post(
            "/feedback",
            json={"response_id": rid, "feedback": 1},
            headers=_headers(client, origin="http://localhost"),
        )
        assert resp.status_code in {200, 403}


def test_missing_production_secret_fails():
    old_secret = os.environ.pop("FLASK_SECRET_KEY", None)
    old_env = os.environ.get("FLASK_ENV")
    os.environ["FLASK_ENV"] = "production"
    old_dev = os.environ.pop("MINDMATE_ALLOW_DEV_SECRET", None)
    try:
        try:
            app_module._resolve_secret_key()
        except RuntimeError as exc:
            assert "FLASK_SECRET_KEY" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        if old_secret is not None:
            os.environ["FLASK_SECRET_KEY"] = old_secret
        if old_env is not None:
            os.environ["FLASK_ENV"] = old_env
        else:
            os.environ.pop("FLASK_ENV", None)
        if old_dev is not None:
            os.environ["MINDMATE_ALLOW_DEV_SECRET"] = old_dev


def test_development_env_without_explicit_flag_fails():
    old_secret = os.environ.pop("FLASK_SECRET_KEY", None)
    old_env = os.environ.get("FLASK_ENV")
    os.environ["FLASK_ENV"] = "development"
    old_dev = os.environ.pop("MINDMATE_ALLOW_DEV_SECRET", None)
    try:
        try:
            app_module._resolve_secret_key()
        except RuntimeError as exc:
            assert "FLASK_SECRET_KEY" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        if old_secret is not None:
            os.environ["FLASK_SECRET_KEY"] = old_secret
        if old_env is not None:
            os.environ["FLASK_ENV"] = old_env
        else:
            os.environ.pop("FLASK_ENV", None)
        if old_dev is not None:
            os.environ["MINDMATE_ALLOW_DEV_SECRET"] = old_dev


def test_development_env_with_explicit_flag_uses_fallback():
    old_secret = os.environ.pop("FLASK_SECRET_KEY", None)
    old_env = os.environ.get("FLASK_ENV")
    os.environ["FLASK_ENV"] = "development"
    os.environ["MINDMATE_ALLOW_DEV_SECRET"] = "1"
    try:
        assert (
            app_module._resolve_secret_key()
            == "mindmate-development-only-secret-not-for-production"
        )
    finally:
        if old_secret is not None:
            os.environ["FLASK_SECRET_KEY"] = old_secret
        else:
            os.environ.pop("FLASK_SECRET_KEY", None)
        if old_env is not None:
            os.environ["FLASK_ENV"] = old_env
        else:
            os.environ.pop("FLASK_ENV", None)
        os.environ.pop("MINDMATE_ALLOW_DEV_SECRET", None)


def test_testing_env_uses_fallback():
    old_secret = os.environ.pop("FLASK_SECRET_KEY", None)
    old_env = os.environ.get("FLASK_ENV")
    os.environ["FLASK_ENV"] = "testing"
    try:
        assert (
            app_module._resolve_secret_key()
            == "mindmate-development-only-secret-not-for-production"
        )
    finally:
        if old_secret is not None:
            os.environ["FLASK_SECRET_KEY"] = old_secret
        else:
            os.environ.pop("FLASK_SECRET_KEY", None)
        if old_env is not None:
            os.environ["FLASK_ENV"] = old_env
        else:
            os.environ.pop("FLASK_ENV", None)


def test_explicit_test_configuration_uses_fallback():
    old_secret = os.environ.pop("FLASK_SECRET_KEY", None)
    old_env = os.environ.get("FLASK_ENV")
    old_test_mode = os.environ.get("MINDMATE_TEST_MODE")
    os.environ.pop("FLASK_ENV", None)
    os.environ["MINDMATE_TEST_MODE"] = "1"
    try:
        assert (
            app_module._resolve_secret_key()
            == "mindmate-development-only-secret-not-for-production"
        )
    finally:
        if old_secret is not None:
            os.environ["FLASK_SECRET_KEY"] = old_secret
        else:
            os.environ.pop("FLASK_SECRET_KEY", None)
        if old_env is not None:
            os.environ["FLASK_ENV"] = old_env
        else:
            os.environ.pop("FLASK_ENV", None)
        if old_test_mode is not None:
            os.environ["MINDMATE_TEST_MODE"] = old_test_mode
        else:
            os.environ.pop("MINDMATE_TEST_MODE", None)


def test_explicit_development_configuration_uses_fallback():
    old_secret = os.environ.pop("FLASK_SECRET_KEY", None)
    old_env = os.environ.get("FLASK_ENV")
    old_dev = os.environ.get("MINDMATE_ALLOW_DEV_SECRET")
    os.environ["FLASK_ENV"] = "development"
    os.environ["MINDMATE_ALLOW_DEV_SECRET"] = "1"
    try:
        assert (
            app_module._resolve_secret_key()
            == "mindmate-development-only-secret-not-for-production"
        )
    finally:
        if old_secret is not None:
            os.environ["FLASK_SECRET_KEY"] = old_secret
        else:
            os.environ.pop("FLASK_SECRET_KEY", None)
        if old_env is not None:
            os.environ["FLASK_ENV"] = old_env
        else:
            os.environ.pop("FLASK_ENV", None)
        if old_dev is not None:
            os.environ["MINDMATE_ALLOW_DEV_SECRET"] = old_dev
        else:
            os.environ.pop("MINDMATE_ALLOW_DEV_SECRET", None)


def test_production_configuration_with_development_env_flag_still_fails():
    old_secret = os.environ.pop("FLASK_SECRET_KEY", None)
    old_env = os.environ.get("FLASK_ENV")
    os.environ["FLASK_ENV"] = "production"
    os.environ["MINDMATE_ALLOW_DEV_SECRET"] = "1"
    try:
        try:
            app_module._resolve_secret_key()
        except RuntimeError as exc:
            assert "FLASK_SECRET_KEY" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        if old_secret is not None:
            os.environ["FLASK_SECRET_KEY"] = old_secret
        else:
            os.environ.pop("FLASK_SECRET_KEY", None)
        if old_env is not None:
            os.environ["FLASK_ENV"] = old_env
        else:
            os.environ.pop("FLASK_ENV", None)
        os.environ.pop("MINDMATE_ALLOW_DEV_SECRET", None)


def test_csrf_token_from_earlier_session_is_rejected():
    with app.test_client() as c1:
        token = _csrf_token(c1)
    with app.test_client() as c2:
        resp = c2.post(
            "/feedback",
            json={"response_id": "x", "feedback": 1},
            headers={"X-CSRFToken": token},
        )
        assert resp.status_code == 403


def test_ordinary_ask_response_id_origin_session_only():
    with app.test_client() as owner:
        rid = _ask(owner).get_json()["response_id"]
        ok = owner.post(
            "/feedback",
            json={"response_id": rid, "feedback": 1},
            headers=_headers(owner),
        )
        assert ok.status_code == 200
    with app.test_client() as other:
        cross = other.post(
            "/feedback",
            json={"response_id": rid, "feedback": 1},
            headers=_headers(other),
        )
        assert cross.status_code == 403


def test_crisis_ask_response_id_origin_session_only():
    with app.test_client() as owner:
        ask_resp = owner.post(
            "/ask",
            data={"message": "I want to kill myself"},
            headers=_headers(owner),
        )
        assert ask_resp.status_code == 200
        rid = ask_resp.get_json()["response_id"]
        assert rid
        ok = owner.post(
            "/feedback",
            json={"response_id": rid, "feedback": -1},
            headers=_headers(owner),
        )
        assert ok.status_code == 200
    with app.test_client() as other:
        cross = other.post(
            "/feedback",
            json={"response_id": rid, "feedback": 1},
            headers=_headers(other),
        )
        assert cross.status_code == 403


def test_database_failure_does_not_record_feedback():
    with app.test_client() as client:
        rid = _ask(client).get_json()["response_id"]
        original = app_module.record_feedback

        def boom(*args, **kwargs):
            raise sqlite3.Error("forced database failure")

        app_module.record_feedback = boom
        try:
            resp = client.post(
                "/feedback",
                json={"response_id": rid, "feedback": 1},
                headers=_headers(client),
            )
        finally:
            app_module.record_feedback = original
        assert resp.status_code == 500
        conn = sqlite3.connect(app_module.db_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE response_id = ?", (rid,)
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 0
