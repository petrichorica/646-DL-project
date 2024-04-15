from PIL import Image
import requests
import faiss
import json
import time
import util

from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class mySigLipModel(nn.Module):
    def __init__(self):
        super(mySigLipModel, self).__init__()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.shape = 768

    def get_image_embedding(self, images):
        self.model.to(device)
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        outputs = self.model.get_image_features(**inputs).cpu()
        return outputs.detach().numpy()

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

group_size = 20

for i in range(start_idx, end_idx, group_size):
    images = []

    for j, img_url in enumerate(train_images[i:i+group_size]):
        img_url = img_url.strip()
        try:
            frame = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        except:
            print(f"Failed to load image {i+j}: {img_url}")
            continue
        images.append(frame)
        image_urls.append(img_url)
    
    embeddings = extractor.get_image_embedding(images)
    index.add(embeddings)

end = time.time()
print(f'Finish indexing for {start_idx} to {end_idx} in ' + str(end - start) + ' seconds')
faiss.write_index(index, util.index_path)

with open(util.image_path, "w") as f:
    json.dump(image_urls, f, indent=2)