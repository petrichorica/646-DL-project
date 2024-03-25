from PIL import Image
import requests
import faiss
import json
from extract_features import mySigLipModel
import time

#load in the embedding extractor
extractor = mySigLipModel()
#load in indexer
index = faiss.IndexFlatL2(extractor.shape)
#load in image dataset
train_images_path = "./dataset/SBU_captioned_photo_dataset_urls.txt"
train_images = open(train_images_path, "r").readlines()
#load in storage path
# image_root = './dataset/paris'
feature_root = './indexed_dataset/'

image_urls = []

print("start indexing...")
start = time.time()

for img_url in train_images[:100]:
    img_url = img_url.strip()
    try:
        frame = Image.open(requests.get(img_url, stream=True).raw)
    except:
        print(f"Failed to load image {img_url}")
        continue
    image_urls.append(img_url)
    embedding = extractor.get_image_embedding(frame)
    # print(embedding.shape)
    index.add(embedding)
    # print(f"Added image {img_url} to index")

end = time.time()
print('Finish in ' + str(end - start) + ' seconds')
faiss.write_index(index, feature_root + "siglip-image-index.bin")

with open(feature_root + "siglip_image_urls.json", "w") as f:
    json.dump(image_urls, f, indent=2)
