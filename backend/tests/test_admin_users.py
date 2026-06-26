"""Testes — gestão de utilizadores do backoffice."""
from __future__ import annotations

import admin_store


def test_admin_can_update_user_role(client):
    headers = {"Authorization": "Bearer test-admin-token"}
    created = admin_store.create_user("pytest_editor", "TestPass123!", "viewer")
    users = admin_store.list_users()
    user_row = next(u for u in users if u["username"] == "pytest_editor")

    res = client.patch(
        f"/api/admin/users/{user_row['id']}",
        json={"role": "editor"},
        headers=headers,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["user"]["role"] == "editor"

    updated = admin_store.list_users()
    row = next(u for u in updated if u["id"] == user_row["id"])
    assert row["role"] == "editor"

    admin_store.delete_user(user_row["id"])


def test_admin_cannot_demote_last_admin(client):
    headers = {"Authorization": "Bearer test-admin-token"}
    users = admin_store.list_users()
    admin_user = next(u for u in users if u["role"] == "admin")

    res = client.patch(
        f"/api/admin/users/{admin_user['id']}",
        json={"role": "editor"},
        headers=headers,
    )
    assert res.status_code == 400
    assert "administrador" in res.json()["detail"].lower()
