# test_api.py
import requests
import json


BASE_URL = 'https://salary-predict-fastapi.azurewebsites.net'  # Replace with your API URL


def test_get():
    response = requests.get(f'{BASE_URL}/')  # Replace with your GET endpoint
    print(response.json())
    assert response.status_code == 200
    # Replace with your expected response
    assert response.json() == {'Hello': 'World'}


def test_post_prediction1():
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
    response = requests.post(f'{BASE_URL}/predict', data=json.dumps(data))
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {'prediction': '<=50K'}

if __name__ == '__main__':
    test_get()
    test_post_prediction1()
