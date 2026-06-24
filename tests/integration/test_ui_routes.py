from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_home_page_returns_200_with_content() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Deteksi & Konsultasi Kanker Kulit" in response.text


def test_upload_page_returns_200_with_content() -> None:
    response = client.get("/upload")
    assert response.status_code == 200
    assert "Upload Gambar" in response.text


def test_chat_page_returns_200_with_content() -> None:
    response = client.get("/chat")
    assert response.status_code == 200
    assert "Chatbot — SkinVision" in response.text
