from PIL import Image
import requests
import torch
from transformers import AutoProcessor, CLIPVisionModel
import faiss
import json

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(frame):
    image = frame.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    features = outputs.last_hidden_state
    features = torch.flatten(features, start_dim=1)
    return features.detach().numpy()

index = faiss.IndexFlatL2(38400)
train_images_path = "./dataset/SBU_captioned_photo_dataset_urls.txt"
train_images = open(train_images_path, "r").readlines()

image_urls = []

for img_url in train_images[:100]:
    img_url = img_url.strip()
    try:
        frame = Image.open(requests.get(img_url, stream=True).raw)
    except:
        print(f"Failed to load image {img_url}")
        continue
    image_urls.append(img_url)
    embedding = get_image_embedding(frame)
    index.add(embedding)
    print(f"Added image {img_url} to index")

faiss.write_index(index, "clip-image-index.bin")

with open("image_urls.json", "w") as f:
    json.dump(image_urls, f, indent=2)
