def test_chat_api_returns_reply_and_units(client):
    response = client.post(
        "/api/chat",
        json={"message": "我喜欢软笔书法，也考过七级", "user_id": "demo_user", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["assistant_message"]
    assert data["session_id"]
    assert data["user_unit_uid"]
    assert data["assistant_unit_uid"]
    assert len(data["created_unit_uids"]) == 2
    assert "retrieved_memories" in data


def test_similar_second_message_stays_in_same_session(client):
    first = client.post(
        "/api/chat",
        json={"message": "我最近在练习软笔书法。", "user_id": "demo_user", "top_k": 3},
    ).json()
    second = client.post(
        "/api/chat",
        json={"message": "书法练习我想继续记录一下。", "user_id": "demo_user", "top_k": 3},
    ).json()

    assert first["session_id"] == second["session_id"]


def test_memory_search_finds_keyword_memory(client):
    client.post(
        "/api/chat",
        json={"message": "我喜欢软笔书法，也考过七级", "user_id": "demo_user", "top_k": 3},
    )

    response = client.post("/api/memory/search", json={"query": "书法", "top_k": 5})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert any("书法" in item["content"] for item in results)

