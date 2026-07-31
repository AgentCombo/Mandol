def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["mandol_ready"] is True
    assert data["llm_mode"] in {"mock", "real"}
    assert "active_session_id" in data


def test_index_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Mandol Chat" in response.text

