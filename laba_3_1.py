import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic_expanded.csv')

def categorize_price(NObeyesdad):
    if NObeyesdad == 'Obesity_Type_I':
        return 1
    elif NObeyesdad == 'Obesity_Type_II':
        return 1
    elif NObeyesdad == 'Obesity_Type_III':
        return 1
    elif NObeyesdad == 'Normal_Weight':
        return 2
    elif NObeyesdad == 'Overweight_Level_I':
        return 2
    elif NObeyesdad == 'Overweight_Level_II':
        return 2
    else:
        return 2

# Добавление новых признаков
data['BMI'] = data['Weight'] / (data['Height'] ** 2)
data['Age_FAF_Ratio'] = data['Age'] / (data['FAF'] + 1)  # +1 для избежания деления на ноль

data['NObeyesdad_cat'] = data['NObeyesdad'].apply(categorize_price)


# OneHotEncoding для категориальных признаков
categorical_columns = data.select_dtypes(include=['object']).columns.drop('NObeyesdad')
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = pd.DataFrame(
    encoder.fit_transform(data[categorical_columns]),
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Объединение новых закодированных признаков с исходными
data = pd.concat([data.drop(columns=categorical_columns), encoded_features], axis=1)

# Выбор признаков для кластеризации
features = data.drop(columns=['NObeyesdad', 'NObeyesdad_cat'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Применение PCA
pca = PCA(n_components=0.5)  # Сохраняем первые 10 компонент
features_pca = pca.fit_transform(features_scaled)
print(f"Объясненная дисперсия после PCA: {np.sum(pca.explained_variance_ratio_):.2f}")

# Тестирование различных методов кластеризации
results = []

# 1. Спектральная кластеризация с разным числом кластеров
for n_clusters in range(2, 3):
    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=1500,  # Увеличено количество соседей для связности графа
        random_state=42
    )
    clusters = spectral_clustering.fit_predict(features_pca)

    silhouette_avg = silhouette_score(features_pca, clusters)
    adjusted_rand = adjusted_rand_score(data['NObeyesdad_cat'], clusters)
    results.append((f"Spectral Clustering (n_clusters={n_clusters})", silhouette_avg, adjusted_rand))

    # Визуализация
    reduced_features = PCA(n_components=2).fit_transform(features_pca)

    plt.figure(figsize=(8, 6))
    for cluster in range(n_clusters):
        cluster_points = reduced_features[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

    plt.title(f"Spectral Clustering (n_clusters={n_clusters})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

# 2. KMeans с разным числом кластеров
for n_clusters in range(2, 3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_pca)

    silhouette_avg = silhouette_score(features_pca, clusters)
    adjusted_rand = adjusted_rand_score(data['NObeyesdad_cat'], clusters)
    results.append((f"KMeans (n_clusters={n_clusters})", silhouette_avg, adjusted_rand))

    # Визуализация
    reduced_features = PCA(n_components=2).fit_transform(features_pca)

    plt.figure(figsize=(8, 6))
    for cluster in range(n_clusters):
        cluster_points = reduced_features[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

    plt.title(f"KMeans (n_clusters={n_clusters})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

# Вывод результатов
print("\nРезультаты кластеризации:")
for method, silhouette_avg, adjusted_rand in results:
    print(f"{method}: Silhouette Score = {silhouette_avg:.2f}, Adjusted Rand Index = {adjusted_rand:.2f}")
