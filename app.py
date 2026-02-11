import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from flask import Flask, render_template, request

app = Flask(__name__)

# Choisis un dossier racine chez l'utilisateur (à adapter)
BASE_PHOTOS_DIR = os.path.join(os.path.expanduser("~"), "Pictures")

ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")


def list_folders(base_dir: str):
    if not os.path.exists(base_dir):
        return []
    folders = []
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            folders.append(name)
    folders.sort()
    return folders


def list_images_in_folder(folder_path: str):
    if not folder_path or not os.path.exists(folder_path):
        return []
    images = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(ALLOWED_EXT) and not f.startswith("kmeans_"):
            images.append(f)
    images.sort()
    return images


def kmeans_color_only(folder_path: str, filename: str, k: int):
    path = os.path.join(folder_path, filename)
    img = Image.open(path).convert("RGB")
    img.thumbnail((500, 500))

    data = np.array(img)
    h, w, _ = data.shape
    pixels = data.reshape(-1, 3).astype(np.float32)

    km = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = km.fit_predict(pixels)

    centers = km.cluster_centers_
    new_pixels = centers[labels].reshape(h, w, 3)
    new_image = np.clip(new_pixels, 0, 255).astype(np.uint8)

    out_name = f"kmeans_k{k}_{os.path.splitext(filename)[0]}.png"
    out_path = os.path.join(folder_path, out_name)
    Image.fromarray(new_image).save(out_path)

    return out_name


@app.route("/")
def home():
    folders = list_folders(BASE_PHOTOS_DIR)
    selected_folder = request.args.get("folder")

    # dossier sélectionné par défaut
    if (not selected_folder) and folders:
        selected_folder = folders[0]

    folder_path = os.path.join(BASE_PHOTOS_DIR, selected_folder) if selected_folder else None
    images = list_images_in_folder(folder_path) if folder_path else []

    selected_img = request.args.get("img")
    algo = request.args.get("algo", "original")
    k = request.args.get("k", 6)

    try:
        k = int(k)
    except:
        k = 6
    k = max(2, min(k, 32))

    if (not selected_img) and images:
        selected_img = images[0]

    display_image = selected_img

    # IMPORTANT : ici on écrit l'image générée dans le dossier photos de l'utilisateur
    if folder_path and selected_img and algo == "kmeans":
        display_image = kmeans_color_only(folder_path, selected_img, k)

    # Pour afficher l'image dans le navigateur, on ne peut afficher que ce que Flask "sert".
    # Donc: on ne peut PAS directement servir un fichier arbitraire du disque sans une route dédiée.
    # On va passer par une route /photo?folder=...&img=...
    return render_template(
        "home.html",
        base_dir=BASE_PHOTOS_DIR,
        folders=folders,
        folder=selected_folder,
        images=images,
        selected=selected_img,
        algo=algo,
        k=k,
        display_image=display_image
    )


@app.route("/photo")
def photo():
    # Sert un fichier depuis BASE_PHOTOS_DIR / folder / img
    from flask import send_file, abort

    folder = request.args.get("folder", "")
    img = request.args.get("img", "")

    # sécurité: on n'autorise que les sous-dossiers de BASE_PHOTOS_DIR
    folder_path = os.path.join(BASE_PHOTOS_DIR, folder)
    if not os.path.isdir(folder_path):
        return abort(404)

    file_path = os.path.join(folder_path, img)
    if not os.path.isfile(file_path):
        return abort(404)

    return send_file(file_path)


if __name__ == "__main__":
    app.run(debug=True, port=5002)