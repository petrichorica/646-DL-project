import faiss
from extract_features import mySigLipModel
from display_image import ImageDisplay
import util
import time

index_path = util.index_path

#load in the embedding extractor
extractor = mySigLipModel()
#load in image displayer
display = ImageDisplay()
#load in indexer
index = faiss.read_index(index_path)

start = time.time()

#load in user query
query = "temples in thailand with blue sky"
query_embedding = extractor.get_text_embedding(query)

#search based on similarity
results_num = 10
D, I = index.search(query_embedding, results_num)
results = I[0]

end = time.time()
print(f"Search time: {end-start} seconds")

#display the results
print(f"Displaying search results for query: {query}")
display.display2(results)