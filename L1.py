import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

data = pd.read_csv('AB_NYC_2019_expanded.csv')
data = data[data['neighbourhood_group'] == 'Brooklyn']

def calculate_slope(x, y):
    mx = x - x.mean()
    my = y - y.mean()
    return sum(mx * my) / sum(mx**2)

def get_params(x, y):
    a = calculate_slope(x, y)
    b = y.mean() - a * x.mean()
    return a, b

d = data[data.price > 0]
x = d.number_of_reviews
y = np.log(d.price)
a, b = get_params(x, y)

lin_reg = a*x + b

print(">>>", a, "\n>>>", b)

plt.xlabel('Количество отзывов')
plt.ylabel('Цена в логарифмическом масштабе')
plt.scatter(x, y)
plt.plot(x, lin_reg, color='red')

plt.show()


X = d.number_of_reviews
y = np.log(d.price)
b, squared_error_sum, matrix_rank, SVD_ = scipy.linalg.lstsq(X, y)






