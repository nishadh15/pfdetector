import requests
from requests.exceptions import ConnectionError, RequestException, Timeout


url = "http://127.0.0.1:8000/predict"
student_data = {
    "study_hours": 5.0,
    "attendance": 80.0,
    "previous_marks": 75.0
}

try:
    response = requests.post(url, json=student_data, timeout=30)
    print("Status code:", response.status_code)
    print("Response:", response.json())
except ConnectionError:
    print("Could not connect to the API. Start it with: uvicorn app:app --reload")
except Timeout:
    print("The API took longer than 30 seconds to respond.")
except RequestException as error:
    print(f"Request failed: {error}")
