# Projet K-Means Clustering

## 📝 Description

Ce projet implémente l’algorithme **K-Means** en Python pour effectuer du **clustering** sur différents jeux de données.  
Il permet de comparer deux méthodes d’initialisation des centroids : `random` et `k-means++`, et d’évaluer la qualité du clustering à l’aide des métriques **Silhouette Score** et **Davies-Bouldin Score**.

### Objectifs pédagogiques :
- Comprendre le fonctionnement de l’algorithme K-Means.  
- Comparer l’impact des différentes méthodes d’initialisation.  
- Évaluer les résultats à l’aide de métriques de clustering.  
- Analyser des datasets réels et tirer des conclusions pertinentes.

---

## 💻 Technologies utilisées

- **Python 3**
- Bibliothèques : `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- Visualisation : `matplotlib.pyplot` pour les graphiques et PCA

---

## 📊 Résultats et analyses par dataset

### 🟢 Dataset 1 : Iris
**Meilleur k : 2**  
**Initialisation : random / k-means++**  

| init      | k | silhouette | davies_bouldin |
| --------- | - | ---------- | --------------- |
| random    | 2 | 0.582      | 0.593           |
| random    | 3 | 0.459      | 0.834           |
| random    | 4 | 0.387      | 0.870           |
| random    | 5 | 0.346      | 0.948           |
| k-means++ | 2 | 0.582      | 0.593           |
| k-means++ | 3 | 0.460      | 0.834           |
| k-means++ | 4 | 0.387      | 0.870           |
| k-means++ | 5 | 0.346      | 0.948           |

*Analyse :*  
- k = 2 donne le meilleur compromis (Silhouette max, Davies-Bouldin min)  
- Résultats très similaires entre random et k-means++  
- Features : sepal length, sepal width, petal length, petal width

---

### 🟣 Dataset 2 : Wine
**Meilleur k : 3**  
**Initialisation : random / k-means++**

| init      | k | silhouette | davies_bouldin |
| --------- | - | ---------- | --------------- |
| random    | 2 | 0.268      | 1.448           |
| random    | 3 | 0.285      | 1.389           |
| random    | 4 | 0.252      | 1.817           |
| random    | 5 | 0.245      | 1.736           |
| k-means++ | 2 | 0.259      | 1.526           |
| k-means++ | 3 | 0.285      | 1.389           |
| k-means++ | 4 | 0.260      | 1.797           |
| k-means++ | 5 | 0.202      | 1.808           |

*Analyse :*  
- k = 3 cohérent avec les 3 classes réelles  
- random et k-means++ donnent résultats identiques pour k=3  
- Features : alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280/od315, proline

---

### 🟡 Dataset 3 : Mall Customers
**Meilleur k : 5**  
**Initialisation : k-means++**

| init      | k | silhouette | davies_bouldin |
| --------- | - | ---------- | --------------- |
| random    | 2 | 0.252      | 1.614           |
| random    | 3 | 0.260      | 1.357           |
| random    | 4 | 0.297      | 1.296           |
| random    | 5 | 0.304      | 1.160           |
| k-means++ | 2 | 0.252      | 1.614           |
| k-means++ | 3 | 0.260      | 1.357           |
| k-means++ | 4 | 0.298      | 1.281           |
| k-means++ | 5 | 0.304      | 1.167           |

*Analyse :*  
- k = 5 logique pour segmenter différents profils clients  
- k-means++ légèrement plus stable  
- Features : Age, Annual Income, Spending Score

---

### 🟣 Dataset 4 : Breast Cancer
**Meilleur k : 2**  
**Initialisation : random / k-means++**

| init      | k | silhouette | davies_bouldin |
| --------- | - | ---------- | --------------- |
| random    | 2 | 0.343      | 1.321           |
| random    | 3 | 0.314      | 1.529           |
| random    | 4 | 0.271      | 1.513           |
| random    | 5 | 0.176      | 1.725           |
| k-means++ | 2 | 0.343      | 1.321           |
| k-means++ | 3 | 0.314      | 1.529           |
| k-means++ | 4 | 0.283      | 1.489           |
| k-means++ | 5 | 0.158      | 1.756           |

*Analyse :*  
- k = 2 correspond aux classes bénin / malin  
- random et k-means++ donnent résultats proches  
- Features : 30 variables (mean radius, mean texture, mean perimeter, …)

---

### ✅ Conclusion générale

| Dataset        | k optimal | Initialisation recommandée   |
| -------------- | --------- | ---------------------------- |
| Iris           | 2         | random / k-means++           |
| Wine           | 3         | random / k-means++           |
| Mall Customers | 5         | k-means++                    |
| Breast Cancer  | 2         | random / k-means++           |

*Synthèse :*  
- Le meilleur k varie selon les datasets  
- Silhouette à maximiser, Davies-Bouldin à minimiser → résultats cohérents  
- k-means++ plus stable et souvent légèrement meilleur

