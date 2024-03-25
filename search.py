from PIL import Image
import requests
import torch
from transformers import AutoProcessor, CLIPVisionModel
import faiss
import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(frame):
    image = frame.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    features = outputs.last_hidden_state
    features = torch.flatten(features, start_dim=1)
    return features.detach().numpy()

query_url = "https://static.flickr.com/2432/3801566410_bca2441029.jpg"
query_image = Image.open(requests.get(query_url, stream=True).raw)
query_embedding = get_image_embedding(query_image)

index = faiss.read_index("clip-image-index.bin")
results_num = 3
D, I = index.search(query_embedding, results_num)

image_urls = open("image_urls.json", "r").read()
train_images = json.loads(image_urls)

for i in I[0]:
    img_url = train_images[i]
    try:
        frame = Image.open(requests.get(img_url, stream=True).raw)
    except:
        print(f"Failed to load image {img_url}")
        continue
    frame.show()
    print(f"Image {img_url} is similar to query image")

# Path: clip-image-index.bin
# This file contains the index of image embeddings created using the clip-image-index.py script.
