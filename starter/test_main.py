# test_bar.py

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_post_predict1():
    data = {
        "age": 39,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_post_predict2():
    data = {
        "age": 39,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}
