from fastapi.testclient import TestClient

from app import CONFIG, app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    client = TestClient(app)
    with open(CONFIG.test_data, 'rb') as f:
        file = {'file': f}
        response = client.post("/predict", files=file)
    assert response.status_code == 200
    assert response.content.decode().split() == ['0', '0', '0', '0',
                                                 '1', '1', '0']


if __name__ == '__main__':
    test_health()
    test_predict()
    print('TESTS OK')
