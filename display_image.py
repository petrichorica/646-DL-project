from PIL import Image
import matplotlib.pyplot as plt
import requests
import json

class ImageDisplay():
    def __init__(self):
        self.image_path = "./indexed_dataset/siglip_image_urls.json"
        self.image_urls = open(self.image_path, "r").read()
        self.train_images = json.loads(self.image_urls)

    def display_image(self, results, query_url, cols, rows):
        query_image = Image.open(requests.get(query_url, stream=True).raw)

        print("Displaying image")
        fig = plt.figure(figsize=(16, 48))
        columns = cols
        rows = rows

        plt.figure()
        query_image = query_image.resize((512,512))
        plt.subplot(rows, columns, 1)
        plt.imshow(query_image)
        plt.grid(False);
        plt.axis('off');

        for index, i in enumerate(results):
            img_url = self.train_images[i]
            try:
                frame = Image.open(requests.get(img_url, stream=True).raw)
            except:
                print(f"Failed to load image {img_url}")
                continue
            img = frame.resize((512,512))
            plt.subplot(rows, columns, index+2)

            plt.imshow(img)
            plt.grid(False);
            plt.axis('off');

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1.8, bottom=0, top=0.8)
        plt.show()

    def display2(self, results):
        #display search results
        for i in results:
            img_url = self.train_images[i]
            try:
                frame = Image.open(requests.get(img_url, stream=True).raw)
            except:
                print(f"Failed to load image {img_url}")
                continue
            frame.show()
            print(f"Image {img_url} is similar to query image")