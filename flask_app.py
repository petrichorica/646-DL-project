from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
import json
from PIL import Image
from extract_features import mySigLipModel
from flask_cors import CORS
from flask_cors import cross_origin
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/static')
CORS(app)


# Limit file upload size to 8MB
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

l2_index_path = './l2_index/0-200k.bin'
image_path_l2 = './l2_index/0-200k.json'

hnsw_index_path = './hnsw_index/0-100k.bin'
image_path_hnsw = './hnsw_index/0-100k.json'

l2_index = faiss.read_index(l2_index_path)
print("Total vectors in the l2 index:", l2_index.ntotal)

with open(image_path_l2, "r") as f:
    image_urls_l2 = json.load(f)
print("Number of URLs loaded:", len(image_urls_l2))

hnsw_index = faiss.read_index(hnsw_index_path)
print("Total vectors in the hnsw index:", hnsw_index.ntotal)

with open(image_path_hnsw, "r") as f:
    image_urls_hnsw = json.load(f)
print("Number of URLs loaded:", len(image_urls_hnsw))

extractor = mySigLipModel()

@app.route('/', methods=['GET'])
def home():
    return render_template('app.html')

@app.route('/search_by_caption', methods=['POST'])
def search_by_caption():
    data = request.json
    caption = data['caption']
    indexing = request.args.get('indexing', 'FlatL2')
    # k = data.get('k', 10)
    k = 50

    print("Caption received:", caption)
    print("Indexing:", indexing)

    caption_embedding = extractor.get_text_embedding(caption)
    # print("Caption Embedding:", caption_embedding)

    query_vector = np.array([caption_embedding]).astype('float32').squeeze()
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    print("Query Vector Shape:", query_vector.shape)

    if indexing == 'FlatL2':
        distances, indices = l2_index.search(query_vector, k)
        result_images = [image_urls_l2[idx] for idx in indices[0]]

    elif indexing == 'HNSW':
        distances, indices = hnsw_index.search(query_vector, k)
        result_images = [image_urls_hnsw[idx] for idx in indices[0]]

    else:
        return jsonify({'error': 'Invalid indexing method'}), 400

    return jsonify({
        'distances': distances.tolist(),
        'images': result_images
    })

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/search_by_image', methods=['POST'])
def search_by_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    k = 50 #request.args.get('k', 10)
    indexing = request.args.get('indexing', 'FlatL2')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    print("File received:", file.filename)
    print("Indexing:", indexing)

    if file and allowed_file(file.filename):
        image = Image.open(file.stream)
        image_embedding = extractor.get_image_embedding(image)
        query_vector = np.array([image_embedding]).astype('float32').squeeze()
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        print("Query Vector Shape:", query_vector.shape)

        if indexing == 'FlatL2':
            distances, indices = l2_index.search(query_vector, k)
            result_images = [image_urls_l2[idx] for idx in indices[0]]
        elif indexing == 'HNSW':
            distances, indices = hnsw_index.search(query_vector, k)
            result_images = [image_urls_hnsw[idx] for idx in indices[0]]
        else:
            return jsonify({'error': 'Invalid indexing method'}), 400

        return jsonify({
            'distances': distances.tolist(),
            'images': result_images
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/save_file', methods=['POST'])
def save_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        # Define the path to the images folder
        save_path = os.path.join(app.root_path, 'images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create the directory if it does not exist
        file.save(os.path.join(save_path, filename))
        return jsonify({'message': 'File saved successfully', 'path': os.path.join(save_path, filename)}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.errorhandler(404)
def page_not_found(e):
    return '''
    <h1>404 Not Found</h1>
    ''', 404


if __name__ == '__main__':
    app.run(debug=False)
