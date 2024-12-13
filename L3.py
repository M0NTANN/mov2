import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('bank_transactions_data_2.csv')

def categorize_price(price):
    if price < 8000:
        return 1
    elif 8000 < price < 18500:
        return 2
    else:
        return 3

# Добавление новых признаков
# Получим текущий год


# Удаление ненужных столбцов
#data = data.drop(columns=["ID", "Model", 'Engine volume', 'Color', 'Doors'])  # Assuming 'Model' is too specific


# OneHotEncoding для категориальных признаков
categorical_columns = data.select_dtypes(include='object').columns
label_encoders = {}
for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Масштабирование данных
scaler = MinMaxScaler()  # Изменено на MinMaxScaler для улучшенной кластеризации
features = data.drop(columns=[ 'TransactionAmount','TransactionDuration', 'LoginAttempts', 'AccountBalance'])
features_scaled = scaler.fit_transform(features)

# Применение PCA
pca = PCA(n_components=4)  # Сохраняем первые 3 компонент
features_pca = pca.fit_transform(features_scaled)
print(f"Объясненная дисперсия после PCA: {np.sum(pca.explained_variance_ratio_):.2f}")

# Тестирование различных методов кластеризации
results = []

# 1. Спектральная кластеризация с разным числом кластеров
for n_clusters in range(3, 4):
    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=10,  # Увеличено количество соседей для связности графа
        random_state=42
    )
    clusters = spectral_clustering.fit_predict(features_pca)

    silhouette_avg = silhouette_score(features_pca, clusters)
    adjusted_rand = adjusted_rand_score(data['TransactionType'], clusters)
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
for n_clusters in range(3,4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_pca)

    silhouette_avg = silhouette_score(features_pca, clusters)
    adjusted_rand = adjusted_rand_score(data['TransactionType'], clusters)
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
