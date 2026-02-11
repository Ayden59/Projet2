import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from flask import Flask, render_template, request

app = Flask(__name__)

IMAGE_FOLDER = os.path.join("static", "images")
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")


def list_images():
    images = []
    if os.path.exists(IMAGE_FOLDER):
        for f in os.listdir(IMAGE_FOLDER):
            if f.lower().endswith(ALLOWED_EXT):
                images.append(f)
    images.sort()
    return images


def apply_kmeans(filename, k):
    path = os.path.join(IMAGE_FOLDER, filename)
    img = Image.open(path).convert("RGB")

    img_small = img.copy()
    img_small.thumbnail((500, 500))

    data = np.array(img_small)
    h, w, _ = data.shape
    pixels = data.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype("uint8")

    new_pixels = centers[labels]
    new_image = new_pixels.reshape(h, w, 3)

    output_name = f"kmeans_{k}_{filename}"
    output_path = os.path.join(IMAGE_FOLDER, output_name)

    Image.fromarray(new_image).save(output_path)

    return output_name


@app.route("/")
def home():
    images = list_images()

    selected = request.args.get("img")
    algo = request.args.get("algo", "original")
    k = request.args.get("k", 6)

    try:
        k = int(k)
    except:
        k = 6

    if not selected and images:
        selected = images[0]

    display_image = selected

    if selected and algo == "kmeans":
        display_image = apply_kmeans(selected, k)

    return render_template(
        "home.html",
        images=images,
        selected=selected,
        algo=algo,
        k=k,
        display_image=display_image
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)