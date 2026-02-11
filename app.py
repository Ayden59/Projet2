import os
from flask import Flask, render_template, request

app = Flask(__name__)

IMAGE_FOLDER = os.path.join("static", "images")
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp")


@app.route("/")
def home():
    images = []
    if os.path.exists(IMAGE_FOLDER):
        for file in os.listdir(IMAGE_FOLDER):
            if file.lower().endswith(ALLOWED_EXT):
                images.append(file)

    images.sort()

    # image sélectionnée via l'url : /?img=xxx.jpg
    selected = request.args.get("img")

    # sécurité : si l'image demandée n'existe pas, on met la 1ère
    if not selected or selected not in images:
        selected = images[0] if images else None

    return render_template("home.html", images=images, selected=selected)


if __name__ == "__main__":
    app.run(debug=True, port=5002)