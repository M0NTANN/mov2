import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('car_price_prediction.csv')
#data = data[data['neighbourhood_group'] == 'Brooklyn' or data['neighbourhood_group'] == 'Manhattan' ]
data['Manufacturer']=np.where(data['Manufacturer'] =='LEXUS', 'Basic', data['Manufacturer'])
data['Manufacturer']=np.where(data['Manufacturer'] =='HONDA', 'Basic', data['Manufacturer'])


count_no_sub = len(data[data['Price']>=15000])
count_sub = len(data[data['Price']<=14999])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)


def categorize_price(price):
    if price < 5000:
        return 1
    elif 5001 <= price <= 10000:
        return 2
    elif 10001 <= price <= 15000:
        return 3
    else:
        return 4

def m3():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix

    data_prepared = data.copy()
    # Convert "Mileage" and "Levy" to numeric values
    data_prepared['Mileage'] = data_prepared['Mileage'].str.replace(' km', '').str.replace(' ', '').astype(float)
    data_prepared['Levy'] = pd.to_numeric(data_prepared['Levy'].str.replace('-', '0'))
    data_prepared['Price_Category'] = data_prepared['Price'].apply(categorize_price)

    # Convert categorical variables to encoded format
    categorical_columns = data_prepared.select_dtypes(include='object').columns
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        data_prepared[column] = le.fit_transform(data_prepared[column])
        label_encoders[column] = le

    # Update features and target
    X = data_prepared.drop(columns=['Price', 'Price_Category'])
    y = data_prepared['Price_Category']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    lr_model = LogisticRegression(solver='saga', max_iter=2000, random_state=42)
    lr_model.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    y_pred = lr_model.predict(X_test)

    # Оценка модели
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr_model.score(X_test, y_test)))
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    print(classification_report(y_test, y_pred, zero_division=0))

def m2():
    from sklearn.metrics import confusion_matrix
    # Убираем ненужные столбцы
    columns_to_drop = ['ID', 'Levy', 'Model', 'Engine volume', 'Color', 'Category', 'Leather interior']
    data_cleaned = data.drop(columns=columns_to_drop)



    # Кодирование целевой переменной (neighbourhood_group)
    label_encoder = LabelEncoder()
    data_cleaned['Manufacturer'] = label_encoder.fit_transform(data_cleaned['Manufacturer'])

    # Кодирование категориальных признаков (например, room_type)
    data_encoded = pd.get_dummies(data_cleaned, columns=['Price'], drop_first=True)

    # Разделение данных на признаки и целевую переменную
    X = data_encoded.drop(columns=['Manufacturer'])
    y = data_encoded['Manufacturer']

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    y_pred = lr_model.predict(X_test)

    # Оценка модели
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr_model.score(X_test, y_test)))
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    print(classification_report(y_test, y_pred))
    logit_roc_auc = roc_auc_score(y_test, lr_model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

m3()


