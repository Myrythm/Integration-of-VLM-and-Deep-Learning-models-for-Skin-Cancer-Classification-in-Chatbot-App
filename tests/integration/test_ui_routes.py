from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_home_page_returns_200_with_content() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Skrining Kanker Kulit" in response.text


def test_upload_page_returns_200_with_content() -> None:
    response = client.get("/upload")
    assert response.status_code == 200
    assert "Upload Foto Lesi" in response.text


def test_chat_page_returns_200_with_content() -> None:
    response = client.get("/chat")
    assert response.status_code == 200
    assert "Konteks Klasifikasi" in response.text
