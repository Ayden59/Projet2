# 🎯 Implémentation et comparaison d’algorithmes de clustering sur images

Application développée en **Python** avec une interface **Flask**, ayant pour objectif principal l’implémentation et l’analyse de deux algorithmes de clustering appliqués à des images.

Ce projet met l’accent sur le travail algorithmique et la comparaison de méthodes de segmentation de couleurs :

- 🔵 **KMeans**
- 🟣 **HClust** (clustering hiérarchique – méthode Ward)

Les images sont traitées à partir du dossier `Pictures` de l’utilisateur, et les résultats sont générés dynamiquement afin de visualiser l’impact des différents algorithmes sur la réduction de palette.

L’interface web sert principalement de support expérimental pour tester les paramètres (notamment le nombre de clusters K).

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
