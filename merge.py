import faiss
import json

def merge_index():
    index_file1 = './indexed_dataset/siglip-image-index-0-1000.bin'
    index_file2 = './indexed_dataset/siglip-image-index-1000-2000.bin'
    index_file3 = './indexed_dataset/siglip-image-index-2000-3000.bin'
    index_file4 = './indexed_dataset/siglip-image-index-3000-6000.bin'
    index_file5 = './indexed_dataset/siglip-image-index-6000-8000.bin'
    index_file6 = './indexed_dataset/siglip-image-index-8000-10000.bin'

    index_files = [index_file1, index_file2, index_file3, index_file4, index_file5, index_file6]

    new_index_file = './indexed_dataset/siglip-image-index-0-10000.bin'

    for index_file in index_files:
        index = faiss.read_index(index_file)
        if 'new_index' not in locals():
            new_index = faiss.IndexFlatL2(index.d)
        new_index.merge_from(index)
        # print(new_index.ntotal)

    faiss.write_index(new_index, new_index_file)

def merge_json():
    json_file1 = './indexed_dataset/siglip_image_urls-0-1000.json'
    json_file2 = './indexed_dataset/siglip_image_urls-1000-2000.json'
    json_file3 = './indexed_dataset/siglip_image_urls-2000-3000.json'
    json_file4 = './indexed_dataset/siglip_image_urls-3000-6000.json'
    json_file5 = './indexed_dataset/siglip_image_urls-6000-8000.json'
    json_file6 = './indexed_dataset/siglip_image_urls-8000-10000.json'

    json_files = [json_file1, json_file2, json_file3, json_file4, json_file5, json_file6]

    new_json_file = './indexed_dataset/siglip_image_urls-0-10000.json'

    image_urls = []

    for json_file in json_files:
        with open(json_file, "r") as f:
            image_urls += json.load(f)

    with open(new_json_file, "w") as f:
        json.dump(image_urls, f, indent=2)

merge_index()
merge_json()