import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import seaborn as sns

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

def met_line_reg():
    d = data[data.price > 0]
    x = d.number_of_reviews
    y = np.log(d.price)
    a, b = get_params(x, y)

    lin_reg = a*x + b

    print("коэффициент наклона: ", a, "\nточка пересечения линии с осью ординат: ", b)

    plt.xlabel('Количество отзывов')
    plt.ylabel('Цена в логарифмическом масштабе')
    plt.scatter(x, y)
    plt.plot(x, lin_reg, color='red')

    plt.show()


def met2():
    d = data[data['price'] > 0]
    d['reviews_per_month'].fillna(0, inplace=True)

    # Features and target
    x = d[['reviews_per_month', 'calculated_host_listings_count', 'number_of_reviews']]
    y = d['price']  # Use price directly without log transformation

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression().fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Calculate error metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print('mse: %.3f, mae: %.3f' % (mse, mae))


met2()






