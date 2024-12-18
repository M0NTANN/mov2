import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('car_price_prediction_numeric_1.csv')

def categorize_price(price):
    if price < 8000:
        return 1
    elif 8000 < price < 18500:
        return 2
    else:
        return 3

# Добавление новых признаков
#data = data_prepared.drop(columns=["ID", "Model", 'Engine volume', 'Color', 'Doors'])  # Assuming 'Model' is too specific
#data['Mileage'] = data['Mileage'].str.replace(' km', '').str.replace(' ', '').astype(float)
#data['Levy'] = pd.to_numeric(data['Levy'].str.replace('-', '0'))

# Получим текущий год
current_year = 2024

# 1. Возраст автомобиля
data["Car Age"] = current_year - data["Prod. year"]

# 2. Пробег на год
# Преобразуем пробег в числовой формат (удаляем "км" и приводим к числу)

data["Mileage per Year"] = data["Mileage"] / data["Car Age"].replace(0, 1)  # избегаем деления на 0

# 3. Объем двигателя в литрах

data = data.drop(columns=["ID", "Model", 'Engine volume', 'Color'])  # Assuming 'Model' is too specific

data['Price_Category'] = data['Price'].apply(categorize_price)


# OneHotEncoding для категориальных признаков
# Convert categorical variables to encoded format
categorical_columns = data.select_dtypes(include='object').columns
label_encoders = {}
for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

#Инициализация энкодера
encoder = OneHotEncoder(sparse_output=False)



# Добавление кодированных признаков в основной DataFrame
#data = pd.concat([data, label_encoders], axis=1)

# Объединение новых закодированных признаков с исходными
#data = pd.concat([data.drop(columns=categorical_columns), label_encoders], axis=1)

# Выбор признаков для кластеризации
features = data.drop(columns=['Manufacturer', 'Mileage', 'Doors'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Применение PCA
pca = PCA(n_components=2)  # Сохраняем первые 3 компонент
features_pca = pca.fit_transform(features_scaled)
print(f"Объясненная дисперсия после PCA: {np.sum(pca.explained_variance_ratio_):.2f}")

# Тестирование различных методов кластеризации
results = []

# 1. Спектральная кластеризация с разным числом кластеров
for n_clusters in range(3, 4):
    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=2000,  # Увеличено количество соседей для связности графа
        random_state=42
    )
    clusters = spectral_clustering.fit_predict(features_pca)

    silhouette_avg = silhouette_score(features_pca, clusters)
    adjusted_rand = adjusted_rand_score(data['Price_Category'], clusters)
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
for n_clusters in range(3, 4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_pca)

    silhouette_avg = silhouette_score(features_pca, clusters)
    adjusted_rand = adjusted_rand_score(data['Price_Category'], clusters)
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
