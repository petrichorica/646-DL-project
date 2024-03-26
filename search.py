from PIL import Image
import requests
import faiss
import os
from extract_features import mySigLipModel
from display_image import ImageDisplay
import util

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
feature_root = util.feature_root
index_path = util.index_path

#load in the embedding extractor
extractor = mySigLipModel()
#load in image displayer
display = ImageDisplay()

#load in user query
query_url = "https://static.flickr.com/2432/3801566410_bca2441029.jpg"
query_image = Image.open(requests.get(query_url, stream=True).raw)
query_embedding = extractor.get_image_embedding(query_image)

#search based on similarity
index = faiss.read_index(index_path)
results_num = 3
D, I = index.search(query_embedding, results_num)
results = I[0]

#display the results
display.display_image(results, query_url, results_num+1, 1)

# Path: clip-image-index.bin
# This file contains the index of image embeddings created using the clip-image-index.py script.
