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

app = Flask(__name__)
CORS(app)


# Limit file upload size to 8MB
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

index_path = 'indexed_0-100k/siglip-image-index-0-10k.bin'
index = faiss.read_index(index_path)
print("Total vectors in the index:", index.ntotal)

image_path = 'indexed_0-100k/siglip_image_urls-0-10k.json'

with open(image_path, "r") as f:
    image_urls = json.load(f)
print("Number of URLs loaded:", len(image_urls))


extractor = mySigLipModel()

@app.route('/search_by_caption', methods=['POST'])
def search_by_caption():
    data = request.json
    caption = data['caption']
    # k = data.get('k', 10)
    k = 50

    print("Caption received:", caption)

    caption_embedding = extractor.get_text_embedding(caption)
    # print("Caption Embedding:", caption_embedding)

    query_vector = np.array([caption_embedding]).astype('float32').squeeze()
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    print("Query Vector Shape:", query_vector.shape)

    distances, indices = index.search(query_vector, k)

    result_images = [image_urls[idx] for idx in indices[0]]

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
    try:
        k = int(k)
    except ValueError:
        return jsonify({'error': 'Invalid value for k'}), 400

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    print("File received:", file.filename)

    if file and allowed_file(file.filename):
        image = Image.open(file.stream)
        image_embedding = extractor.get_image_embedding(image)
        query_vector = np.array([image_embedding]).astype('float32').squeeze()
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        print("Query Vector Shape:", query_vector.shape)

        distances, indices = index.search(query_vector, k)

        result_images = [image_urls[idx] for idx in indices[0]]

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
    app.run(debug=True)
