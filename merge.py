import faiss
import json
import os

root = './indexed_100k-200k'

def merge_index():
    index_files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.endswith('.bin')]

    new_index_file = './indexed_100k-200k/siglip-image-index-100k-200k.bin'

    for index_file in index_files:
        file_path = os.path.join(root, index_file)
        index = faiss.read_index(file_path)
        if 'new_index' not in locals():
            new_index = faiss.IndexFlatL2(index.d)
        new_index.merge_from(index)
        # print(new_index.ntotal)

    faiss.write_index(new_index, new_index_file)

def merge_json():
    json_files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.endswith('.json')]

    new_json_file = './indexed_100k-200k/siglip_image_urls-100k-200k.json'

    image_urls = []

    for json_file in json_files:
        file_path = os.path.join(root, json_file)
        with open(file_path, "r") as f:
            image_urls += json.load(f)

    with open(new_json_file, "w") as f:
        json.dump(image_urls, f, indent=2)

merge_index()
merge_json()