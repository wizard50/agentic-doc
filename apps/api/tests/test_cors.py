"""CORS configuration tests."""

from fastapi.testclient import TestClient


def test_cors_allows_configured_origin(client: TestClient) -> None:
    origin = "http://localhost:5173"
    response = client.options(
        "/health",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin


def test_cors_exposes_request_id(client: TestClient) -> None:
    origin = "http://localhost:5173"
    response = client.get("/health", headers={"Origin": origin})
    assert response.status_code == 200
    expose = response.headers.get("access-control-expose-headers", "")
    assert "x-request-id" in expose.lower()
