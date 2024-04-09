feature_root = './indexed_30k-60k/'

start_idx = 55000
end_idx = 60000

index_path = feature_root + "siglip-image-index-{}k-{}k.bin".format(start_idx//1000, end_idx//1000)
image_path = feature_root + "siglip_image_urls-{}k-{}k.json".format(start_idx//1000, end_idx//1000)

query_url = "https://static.flickr.com/2432/3801566410_bca2441029.jpg"
train_images_path = "./dataset/SBU_captioned_photo_dataset_urls.txt"

evaluation_path = "./evaluation/sample-10000/"