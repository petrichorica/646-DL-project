from PIL import Image
import requests
import faiss
import os
import json
from extract_features import mySigLipModel

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#load in the embedding extractor
extractor = mySigLipModel()

#load in user query
query_url = "https://static.flickr.com/2432/3801566410_bca2441029.jpg"
query_image = Image.open(requests.get(query_url, stream=True).raw)
query_embedding = extractor.get_image_embedding(query_image)

#search based on similarity
index = faiss.read_index("clip-image-index.bin")
results_num = 3
D, I = index.search(query_embedding, results_num)

image_urls = open("image_urls.json", "r").read()
train_images = json.loads(image_urls)

#display search results
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
