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
            if f.lower().endswith(ALLOWED_EXT) and not f.startswith("kmeans_"):
                images.append(f)
    images.sort()
    return images


def apply_kmeans_color_position(filename: str, k: int, pos_weight: float = 0.4) -> str:
    """
    KMeans sur [R,G,B,x,y] pour faire une segmentation qui tient compte
    à la fois de la couleur et de la position.
    pos_weight contrôle l'importance de la position (0 = couleur seule).
    """
    path = os.path.join(IMAGE_FOLDER, filename)
    img = Image.open(path).convert("RGB")

    # Réduction pour que ça reste rapide (TP-friendly)
    img_small = img.copy()
    img_small.thumbnail((500, 500))

    data = np.array(img_small)
    h, w, _ = data.shape

    # Couleurs en float
    rgb = data.reshape(-1, 3).astype(np.float32)  # (N,3)

    # Coordonnées (x,y) normalisées entre 0 et 1
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx.reshape(-1, 1).astype(np.float32)) / max(1, (w - 1))
    y = (yy.reshape(-1, 1).astype(np.float32)) / max(1, (h - 1))

    # On met la position à une "échelle" comparable aux couleurs
    # (couleurs ~ 0..255, positions ~ 0..1)
    # Donc on multiplie la position par 255 et par pos_weight.
    xy = np.hstack([x, y]) * 255.0 * float(pos_weight)

    # Features finales : [R, G, B, x, y]
    features = np.hstack([rgb, xy])  # (N,5)

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(features)

    # Pour reconstruire une image visible, on remplace chaque pixel par la couleur moyenne du cluster
    # (on calcule la moyenne des couleurs RGB par cluster)
    new_rgb = np.zeros_like(rgb)

    for cluster_id in range(k):
        mask = labels == cluster_id
        if np.any(mask):
            new_rgb[mask] = np.mean(rgb[mask], axis=0)

    new_rgb = np.clip(new_rgb, 0, 255).astype(np.uint8)
    new_image = new_rgb.reshape(h, w, 3)

    output_name = f"kmeans_xy_k{k}_{os.path.splitext(filename)[0]}.png"
    output_path = os.path.join(IMAGE_FOLDER, output_name)

    if not os.path.exists(output_path):
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

    k = max(2, min(k, 32))

    if (not selected) and images:
        selected = images[0]

    display_image = selected

    if selected and algo == "kmeans":
        # Segmentation couleur + position
        display_image = apply_kmeans_color_position(selected, k, pos_weight=0.4)

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