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
query_url = "http://static.flickr.com/2291/2252298370_3c38a5f6f5.jpg"
query_image = Image.open(requests.get(query_url, stream=True).raw)
query_embedding = extractor.get_image_embedding(query_image)

#search based on similarity
index = faiss.read_index(index_path)
results_num = 5
D, I = index.search(query_embedding, results_num)
results = I[0]

#display the results
display.display3(query_image, results)

# Path: clip-image-index.bin
# This file contains the index of image embeddings created using the clip-image-index.py script.
