"""Admin API auth (F4).

/admin is a PII surface (session transcripts, caller phones). It must refuse
requests without a configured admin_token.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from callmind.admin.store import BusinessStore
from callmind.config import Settings
from callmind.gateway.app import app


@pytest.fixture
def client(tmp_path) -> TestClient:
    app.state.settings = Settings(
        admin_token="adminsecret",
        memory_db_path=str(tmp_path / "callmind.db"),
        kb_dir=str(tmp_path / "kb"),
    )
    app.state.business_store = BusinessStore(str(tmp_path / "callmind.db"))
    return TestClient(app)


def test_admin_requires_token(client):
    r = client.get("/admin/businesses")
    assert r.status_code == 401


def test_admin_rejects_bad_token(client):
    r = client.get("/admin/businesses", headers={"x-admin-token": "nope"})
    assert r.status_code == 401


def test_admin_accepts_valid_token(client):
    r = client.get("/admin/businesses", headers={"x-admin-token": "adminsecret"})
    assert r.status_code == 200


def test_admin_write_needs_token(client):
    r = client.post("/admin/businesses", json={"name": "T"})
    assert r.status_code == 401


def test_admin_not_enabled_without_configured_token(tmp_path):
    app.state.settings = Settings(
        admin_token="",
        memory_db_path=str(tmp_path / "callmind.db"),
        kb_dir=str(tmp_path / "kb"),
    )
    app.state.business_store = BusinessStore(str(tmp_path / "callmind.db"))
    c = TestClient(app)
    r = c.get("/admin/businesses", headers={"x-admin-token": "whatever"})
    assert r.status_code == 503