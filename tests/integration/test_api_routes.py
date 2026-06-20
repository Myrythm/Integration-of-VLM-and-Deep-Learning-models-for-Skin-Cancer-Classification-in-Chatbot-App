from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_upload_image_classifies_lesion():
    # Create a small dummy image (1x1 PNG)
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color="red").save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/upload",
        files={"file": ("test.png", buf, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "detection" in body
    assert "chat_session_id" in body
