from PIL import Image
import requests
import faiss
import json
from extract_features import mySigLipModel
import time
import util

#load in the embedding extractor
extractor = mySigLipModel()
#load in indexer
index = faiss.IndexFlatL2(extractor.shape)
#load in image dataset
# train_images_path = "./dataset/SBU_captioned_photo_dataset_urls.txt"
train_images = open(util.train_images_path, "r").readlines()
#load in storage path
feature_root = util.feature_root
#load in start idx and end idx
start_idx = util.start_idx
end_idx = util.end_idx

image_urls = []

print(f"start indexing for {start_idx} to {end_idx} ...")
start = time.time()

for i, img_url in enumerate(train_images[util.start_idx:util.end_idx]):
    img_url = img_url.strip()
    try:
        frame = Image.open(requests.get(img_url, stream=True).raw)
    except:
        print(f"Failed to load image {i}: {img_url}")
        continue
    image_urls.append(img_url)
    embedding = extractor.get_image_embedding(frame)
    # print(embedding.shape)
    index.add(embedding)

end = time.time()
print(f'Finish indexing for {start_idx} to {end_idx} in ' + str(end - start) + ' seconds')
faiss.write_index(index, util.index_path)

with open(util.image_path, "w") as f:
    json.dump(image_urls, f, indent=2)
