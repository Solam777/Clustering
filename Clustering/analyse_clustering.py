# Importation des bibliothèques nécessaires
import pandas as pd  # Pour manipuler les données sous forme de tableaux (DataFrame)
import numpy as np  # Pour les opérations mathématiques
from sklearn.datasets import load_iris, load_wine, load_breast_cancer  # Jeux de données intégrés
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Prétraitement des données
from sklearn.cluster import KMeans  # Algorithme K-Means
from sklearn.metrics import silhouette_score, davies_bouldin_score  # Métriques d’évaluation de clustering
from sklearn.decomposition import PCA  # Réduction de dimension (pour la visualisation)
import matplotlib.pyplot as plt  # Tracé de graphiques
import seaborn as sns  # Visualisation avec un style plus agréable

# Chargement de plusieurs jeux de données
iris = load_iris(as_frame=True).frame
wine = load_wine(as_frame=True).frame
cancer = load_breast_cancer(as_frame=True).frame
mall_df = pd.read_csv("Mall_Customers.csv")  # Chargement d’un fichier CSV local

# Fonction de visualisation des clusters en 2D après réduction PCA
def plot_clusters_2D(X, labels, centers, name):
    pca = PCA(n_components=2)  # Réduction à 2 dimensions pour affichage
    X_reduced = pca.fit_transform(X)  # Transformation des données
    centers_reduced = pca.transform(centers)  # Transformation des centres

    # Affichage des clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=labels, palette='tab10', s=60)
    plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='black', s=200, alpha=0.6, marker='X',
                label='Centre')  # Ajout des centres des clusters
    plt.title(f"Clusters visualisés (PCA) – {name}")
    plt.legend()
    plt.show()

# Fonction d’analyse complète d’un dataset
def analyze_dataset(name, df, drop_columns=[], label_encode_cols=[]):
    print(f"\n=== Dataset: {name} ===")

    # Suppression des colonnes inutiles
    df_clean = df.drop(columns=drop_columns)

    # Encodage des colonnes catégorielles en valeurs numériques
    for col in label_encode_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])

    # Normalisation des données (centrage-réduction)
    X = StandardScaler().fit_transform(df_clean)

    # Initialisation des variables de suivi
    results = []
    best_score = -1
    best_k = None
    best_init = None
    best_labels = None
    best_centers = None

    # Boucle sur différentes méthodes d'initialisation et valeurs de k (nombre de clusters)
    for init_method in ['random', 'k-means++']:
        for k in range(2, 6):
            # Application de KMeans
            kmeans = KMeans(n_clusters=k, init=init_method, n_init=10, random_state=42)
            labels = kmeans.fit_predict(X)  # Prédiction des clusters

            # Évaluation des performances avec deux métriques
            silhouette = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)

            # Stockage des résultats
            results.append({
                'init': init_method,
                'k': k,
                'silhouette': silhouette,
                'davies_bouldin': db_score,
            })

            # Mise à jour du meilleur score trouvé
            if silhouette > best_score:
                best_score = silhouette
                best_k = k
                best_init = init_method
                best_labels = labels
                best_centers = kmeans.cluster_centers_

    # Affichage des résultats sous forme de DataFrame
    result_df = pd.DataFrame(results)
    print(result_df)

    # Création de graphiques pour analyser les métriques en fonction de k
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    sns.lineplot(data=result_df, x='k', y='silhouette', hue='init', marker='o', ax=axs[0])
    axs[0].set_title('Silhouette Score')
    sns.lineplot(data=result_df, x='k', y='davies_bouldin', hue='init', marker='o', ax=axs[1])
    axs[1].set_title('Davies-Bouldin Score')

    plt.suptitle(f"Clustering Metrics for {name}", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Affichage du meilleur clustering trouvé
    print(f"--> Meilleur k = {best_k} avec init = {best_init} (Silhouette = {best_score:.3f})")
    plot_clusters_2D(X, best_labels, best_centers, name)  # Visualisation des clusters

# Analyse de chacun des jeux de données
analyze_dataset("Iris", iris, drop_columns=["target"])
analyze_dataset("Wine", wine, drop_columns=["target"])
analyze_dataset("Cancer", cancer, drop_columns=["target"])
analyze_dataset("Mall Customers", mall_df, drop_columns=["CustomerID"], label_encode_cols=["Genre"])
