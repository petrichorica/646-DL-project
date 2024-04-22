
import requests

url = 'http://127.0.0.1:5000/search_by_caption'
data = {
    'caption': " Dan, waiting for fish to jump in the boat so he can retake the lead from the NG",
    'k': 10
}
print(data["caption"])
response = requests.post(url, json=data)
print(response.json())