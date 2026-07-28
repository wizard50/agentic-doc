"""Request ID middleware tests."""

from fastapi.testclient import TestClient

from agentic_doc_api.middleware.request_id import REQUEST_ID_HEADER


def test_request_id_generated_when_missing(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert REQUEST_ID_HEADER in response.headers
    assert response.headers[REQUEST_ID_HEADER]


def test_request_id_echoed(client: TestClient) -> None:
    custom_id = "test-request-id-123"
    response = client.get("/health", headers={REQUEST_ID_HEADER: custom_id})
    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == custom_id
