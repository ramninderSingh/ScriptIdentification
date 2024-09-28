import argparse
import requests

parser = argparse.ArgumentParser(description="Send image and model name to FastAPI endpoint.")
parser.add_argument('image_path', type=str, help="Path to the image file")
parser.add_argument('model_name', type=str, help="Model name")

args = parser.parse_args()

url = "http://127.0.0.1:8000/predict/"
files = {'file': open(args.image_path, 'rb')}
data = {'model_name': args.model_name}

response = requests.post(url, files=files, data=data)
print(response.json())
