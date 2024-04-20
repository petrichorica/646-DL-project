feature_root = './hnsw_index_0-100k/'

start_idx = 0
end_idx = 10000

index = 'HNSWFlat'
# index = 'FlatL2'

if (index == 'HNSWFlat'):
    index_path = feature_root + "hnsw_index-{}k-{}k.bin".format(start_idx//1000, end_idx//1000)
    image_path = feature_root + "hnsw_urls-{}k-{}k.json".format(start_idx//1000, end_idx//1000)
elif (index == 'FlatL2'):
    index_path = feature_root + "siglip-image-index-{}k-{}k.bin".format(start_idx//1000, end_idx//1000)
    image_path = feature_root + "siglip_image_urls-{}k-{}k.json".format(start_idx//1000, end_idx//1000)

query_url = "http://static.flickr.com/1126/5157409353_805483d0e4.jpg"
train_images_path = "./dataset/SBU_captioned_photo_dataset_urls.txt"

evaluation_path = "./evaluation/sample-10000/"