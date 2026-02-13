import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from flask import Flask, render_template, request, send_file, abort

app = Flask(__name__)

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
        if f.lower().endswith(ALLOWED_EXT) and not (
            f.startswith("kmeans_") or f.startswith("hclust_") or f.startswith("dbscan_")
        ):
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

def hclust_color_only(folder_path: str, filename: str, k: int):

    path = os.path.join(folder_path, filename)
    img = Image.open(path).convert("RGB")
    img.thumbnail((350, 350))  # limite taille

    data = np.array(img, dtype=np.uint8)
    h, w, _ = data.shape

    pixels = data.reshape(-1, 3).astype(np.float32)
    n = pixels.shape[0]

    sample_size = 8000

    rng = np.random.default_rng(0)

    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    sample_labels = model.fit_predict(sample)

    centers = np.zeros((k, 3), dtype=np.float32)

    for i in range(k):
        mask = (sample_labels == i)
        if np.any(mask):
            centers[i] = sample[mask].mean(axis=0)
        else:
            centers[i] = sample.mean(axis=0)

    d2 = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)

    new_pixels = centers[labels].reshape(h, w, 3)
    new_image = np.clip(new_pixels, 0, 255).astype(np.uint8)

    out_name = f"hclust_k{k}_{os.path.splitext(filename)[0]}.png"
    out_path = os.path.join(folder_path, out_name)
    Image.fromarray(new_image).save(out_path)

    return out_name

# ------------------ DBSCAN ------------------
def dbscan_color_only(
    folder_path: str,
    filename: str,
    min_samples: int,
    eps: float,
    selected_class: int | None = None,
):

    path = os.path.join(folder_path, filename)
    img = Image.open(path).convert("RGB")
    img.thumbnail((350, 350))

    data = np.array(img, dtype=np.uint8)
    h, w, _ = data.shape

    pixels = data.reshape(-1, 3).astype(np.float32)

    model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    labels = model.fit_predict(pixels)

    unique = set(labels.tolist())
    classes = sorted([c for c in unique if c != -1])
    n_clusters = len(classes)
    n_noise = int(np.sum(labels == -1))

    suffix = ""
    if selected_class is not None:
        suffix = f"_class{selected_class}"

    out_name = f"dbscan_ms{min_samples}_eps{str(eps).replace('.','p')}{suffix}_{os.path.splitext(filename)[0]}.png"
    out_path = os.path.join(folder_path, out_name)

    if n_clusters == 0:
        Image.fromarray(data).save(out_path)
        return out_name, 0, [], n_noise

    centers = {}
    for c in classes:
        mask = labels == c
        centers[c] = pixels[mask].mean(axis=0)

    # Affichage filtré (une seule classe) ou toutes
    new_pixels = np.zeros_like(pixels)  # fond noir

    if selected_class is not None:
        if selected_class == -1:
            # Affiche uniquement le bruit
            mask = labels == -1
            new_pixels[mask] = np.array([255, 255, 255], dtype=np.float32)
        elif selected_class in classes:
            mask = labels == selected_class
            new_pixels[mask] = centers[selected_class]
        # si selected_class invalide -> on laisse l'image noire
    else:
        for c in classes:
            mask = labels == c
            new_pixels[mask] = centers[c]

    new_image = np.clip(new_pixels.reshape(h, w, 3), 0, 255).astype(np.uint8)
    Image.fromarray(new_image).save(out_path)

    return out_name, n_clusters, classes, n_noise
# ---------------- FIN DBSCAN ----------------

def safe_subfolder(base: str, folder: str) -> str | None:
    if not folder:
        return None
    base_abs = os.path.abspath(base)
    target_abs = os.path.abspath(os.path.join(base_abs, folder))
    if os.path.commonpath([base_abs, target_abs]) != base_abs:
        return None
    if not os.path.isdir(target_abs):
        return None
    return target_abs

@app.route("/")
def home():
    folders = list_folders(BASE_PHOTOS_DIR)

    selected_folder = request.args.get("folder", "")
    folder_path = safe_subfolder(BASE_PHOTOS_DIR, selected_folder)

    images = list_images_in_folder(folder_path) if folder_path else []

    selected_img = request.args.get("img", "")
    algo = request.args.get("algo", "original")
    k_raw = request.args.get("k", "6")
    linkage = request.args.get("linkage", "ward")

    min_samples_raw = request.args.get("min_samples", "6")
    eps_raw = request.args.get("eps", "18.0")

    selected_class_raw = request.args.get("selected_class", "")
    try:
        selected_class = int(selected_class_raw) if selected_class_raw != "" else None
    except ValueError:
        selected_class = None

    try:
        k = int(k_raw)
    except ValueError:
        k = 6
    k = max(2, min(k, 32))

    try:
        min_samples = int(min_samples_raw)
    except ValueError:
        min_samples = 6
    min_samples = max(2, min(min_samples, 64))

    try:
        eps = float(eps_raw)
    except ValueError:
        eps = 18.0
    eps = max(0.1, min(eps, 200.0))

    if (not selected_img) and images:
        selected_img = images[0]

    display_image = selected_img

    n_clusters = None
    classes = None
    n_noise = None

    if folder_path and selected_img:
        if algo == "kmeans":
            display_image = kmeans_color_only(folder_path, selected_img, k)
        elif algo == "hclust":
            if linkage not in ("ward", "average", "complete", "single"):
                linkage = "ward"
            display_image = hclust_color_only(folder_path, selected_img, k)
        elif algo == "dbscan":
            display_image, n_clusters, classes, n_noise = dbscan_color_only(
                folder_path,
                selected_img,
                min_samples=min_samples,
                eps=eps,
                selected_class=selected_class,
            )

    return render_template(
        "home.html",
        base_dir=BASE_PHOTOS_DIR,
        folders=folders,
        folder=selected_folder,
        images=images,
        selected=selected_img,
        algo=algo,
        k=k,
        display_image=display_image,
        min_samples=min_samples,
        eps=eps,
        n_clusters=n_clusters,
        classes=classes,
        n_noise=n_noise,
        selected_class=selected_class,
    )

@app.route("/photo")
def photo():
    folder = request.args.get("folder", "")
    img = request.args.get("img", "")

    folder_path = safe_subfolder(BASE_PHOTOS_DIR, folder)
    if not folder_path:
        return abort(404)

    file_path = os.path.join(folder_path, img)
    if not os.path.isfile(file_path):
        return abort(404)

    return send_file(file_path)

if __name__ == "__main__":
    app.run(debug=True, port=5002)