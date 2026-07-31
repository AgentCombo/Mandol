def test_sessions_finalize_and_reset(client):
    chat = client.post(
        "/api/chat",
        json={"message": "今天上午我在整理项目计划。", "user_id": "demo_user", "top_k": 3},
    ).json()
    session_id = chat["session_id"]

    sessions = client.get("/api/sessions")
    assert sessions.status_code == 200
    data = sessions.json()
    assert data["all_sessions"]
    assert data["active_session_id"] == session_id

    finalized = client.post(f"/api/sessions/{session_id}/finalize")
    assert finalized.status_code == 200
    assert finalized.json()["session"]["is_finalized"] is True

    after_finalize = client.get("/api/sessions").json()
    matching = [s for s in after_finalize["all_sessions"] if s["session_id"] == session_id]
    assert matching and matching[0]["is_finalized"] is True

    reset = client.post("/api/reset")
    assert reset.status_code == 200
    after_reset = client.get("/api/sessions").json()
    assert after_reset["all_sessions"] == []


def test_build_memory_is_explicit_and_mock_safe(client):
    chat = client.post(
        "/api/chat",
        json={"message": "请记录：我正在测试显式高阶记忆构建。", "user_id": "demo_user", "top_k": 3},
    ).json()

    response = client.post(
        "/api/memory/build",
        json={"session_id": chat["session_id"], "sample_id": "demo_user"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "mock_build"

