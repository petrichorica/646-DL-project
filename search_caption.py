import faiss
from extract_features import mySigLipModel
from display_image import ImageDisplay
import util

index_path = util.index_path

#load in the embedding extractor
extractor = mySigLipModel()
#load in image displayer
display = ImageDisplay()

#load in user query
query = "a dog playing in the park"
query_embedding = extractor.get_text_embedding(query)

#search based on similarity
index = faiss.read_index(index_path)
results_num = 5
D, I = index.search(query_embedding, results_num)
results = I[0]

#display the results
print(f"Displaying search results for query: {query}")
display.display2(results)