# 🖼️ Image Clustering Web App

Application web développée en **Python / Flask** permettant d’appliquer des algorithmes de **clustering de couleurs** sur des images locales.

Le projet permet de comparer deux méthodes de segmentation :

- 🔵 **KMeans**
- 🟣 **HClust** (clustering hiérarchique – méthode Ward)

Les images sont traitées directement depuis le dossier `Pictures` de l'utilisateur et les résultats sont générés dynamiquement.

---

## 🚀 Fonctionnalités

- 📁 Sélection d’un dossier local
- 🖼️ Sélection d’une image
- 🎨 Quantification des couleurs via :
  - KMeans
  - HClust (AgglomerativeClustering - Ward)
- 🔢 Choix du nombre de clusters (**K**)
- 💾 Génération automatique de l’image traitée

---

## 🧠 Algorithmes utilisés

### 🔵 KMeans

- Partitionnement des pixels en **K groupes**
- Chaque pixel est remplacé par le centre de son cluster
- Algorithme rapide et efficace pour la réduction de palette

### 🟣 HClust (Hierarchical Clustering)

- Clustering hiérarchique basé sur la méthode **Ward**
- Réalisé sur un échantillon de pixels afin d’éviter l’explosion mémoire
- Les centres sont calculés puis appliqués à toute l’image

---

## 🛠️ Technologies

- Python 3  
- Flask  
- NumPy  
- Pillow (PIL)  
- scikit-learn  
- HTML / CSS 
