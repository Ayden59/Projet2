import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from flask import Flask, render_template, request

app = Flask(__name__)

IMAGE_FOLDER = os.path.join("static", "images")
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")


def list_images():
    if not os.path.exists(IMAGE_FOLDER):
        return []
    images = []
    for f in os.listdir(IMAGE_FOLDER):
        if f.lower().endswith(ALLOWED_EXT) and not f.startswith("kmeans_"):
            images.append(f)
    images.sort()
    return images


def kmeans_color_position(filename, k, pos_weight=0.4):
    # 1) Lire l'image + réduire (rapide)
    path = os.path.join(IMAGE_FOLDER, filename)
    img = Image.open(path).convert("RGB")
    img.thumbnail((500, 500))

    data = np.array(img)         # (h,w,3)
    h, w, _ = data.shape

    # 2) Construire les features [R,G,B,x,y]
    rgb = data.reshape(-1, 3).astype(np.float32)  # N x 3

    yy, xx = np.mgrid[0:h, 0:w]                   # grilles de coordonnées
    x = (xx.reshape(-1, 1) / max(1, w - 1)).astype(np.float32)  # N x 1 (0..1)
    y = (yy.reshape(-1, 1) / max(1, h - 1)).astype(np.float32)  # N x 1 (0..1)

    xy = np.hstack([x, y]) * 255.0 * pos_weight   # mise à l'échelle
    features = np.hstack([rgb, xy])               # N x 5

    # 3) KMeans + reconstruction (couleurs des centres)
    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = km.fit_predict(features)             # N

    centers_rgb = km.cluster_centers_[:, :3]      # K x 3 (on garde juste RGB)
    new_rgb = centers_rgb[labels].reshape(h, w, 3)
    new_rgb = np.clip(new_rgb, 0, 255).astype(np.uint8)

    # 4) Sauvegarde
    out_name = f"kmeans_xy_k{k}_{os.path.splitext(filename)[0]}.png"
    out_path = os.path.join(IMAGE_FOLDER, out_name)
    Image.fromarray(new_rgb).save(out_path)

    return out_name


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
    k = max(2, min(k, 32))

    if (not selected) and images:
        selected = images[0]

    display_image = selected

    if selected and algo == "kmeans":
        display_image = kmeans_color_position(selected, k, pos_weight=0.4)

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