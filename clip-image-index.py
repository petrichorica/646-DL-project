from PIL import Image
import requests
import faiss
import json
from extract_features import mySigLipModel

#load in the embedding extractor
extractor = mySigLipModel()
#load in indexer
index = faiss.IndexFlatL2(extractor.shape)
#load in image dataset
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
    embedding = extractor.get_image_embedding(frame)
    index.add(embedding)
    print(f"Added image {img_url} to index")

faiss.write_index(index, "clip-image-index.bin")

with open("image_urls.json", "w") as f:
    json.dump(image_urls, f, indent=2)
