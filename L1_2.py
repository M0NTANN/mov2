import pandas as pd
import numpy as np

# Загрузка исходного датасета
data = pd.read_csv('car_price_prediction.csv')


# Функция для увеличения количества записей
def augment_and_expand_dataset(data, augmentations=2, noise_level=0.1):
    augmented_data = []

    for _ in range(augmentations):
        # Копируем данные
        temp_data = data.copy()

        # Добавляем шум к цене
        noise = temp_data['Price'] * np.random.uniform(-noise_level, noise_level, size=len(temp_data))
        temp_data['Price'] += noise

        # Добавляем шум к координатам
        #lat_noise = np.random.uniform(-0.01, 0.01, size=len(temp_data))
        #long_noise = np.random.uniform(-0.01, 0.01, size=len(temp_data))
        #temp_data['Mileage'] += lat_noise
        #temp_data['Mileage'] += long_noise

        # Изменяем текстовые данные
        #temp_data['name'] = temp_data['name'].fillna('Unknown') + " (Augmented)"
        #temp_data['host_name'] = temp_data['host_name'].fillna('Unknown') + np.random.choice(
        #    ['_A', '_B', '_C'], size=len(temp_data))
        temp_data['Manufacturer'] = temp_data['Manufacturer'] + np.random.choice(
            ['LEXUS', 'HONDA', 'HYUNDAI', 'TOYOTA'], size=len(temp_data))
        # Изменяем отзывы
        #temp_data['number_of_reviews'] = temp_data['number_of_reviews'] + np.random.randint(-5, 5, size=len(temp_data))
        #temp_data['number_of_reviews'] = temp_data['number_of_reviews'].clip(lower=0)  # Отрицательные значения в 0

        temp_data['Airbags'] = temp_data['Airbags'] + np.random.randint(-5, 5, size=len(temp_data))
        temp_data['Airbags'] = temp_data['Airbags'].clip(lower=0)  # Отрицательные значения в 0
        augmented_data.append(temp_data)

    # Объединяем оригинальный и аугментированный датасет
    expanded_data = pd.concat([data] + augmented_data, ignore_index=True)
    return expanded_data


# Применение аугментации
expanded_data = augment_and_expand_dataset(data, augmentations=3)

# Сохранение в новый файл
expanded_data.to_csv('car_price_prediction_expanded.csv', index=False)
print("Расширенный датасет сохранен в файл 'car_price_prediction_expanded.csv'.")
