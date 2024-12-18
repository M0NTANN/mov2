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
#data['Manufacturer']=np.where(data['Manufacturer'] =='LEXUS', 'Basic', data['Manufacturer'])
#data['Manufacturer']=np.where(data['Manufacturer'] =='HONDA', 'Basic', data['Manufacturer'])


count_7000 = len(data[data['Price']<8000])
count_7000_20000 = len(data[(data['Price'] > 8000) & (data['Price'] < 18500)])
count_20000 = len(data[data['Price']>18500])
pct_of_7000 = count_7000/(count_7000+count_7000_20000+count_20000)
print("percentage of no subscription is", pct_of_7000*100)
pct_of_7000_20000 = count_7000_20000/(count_7000+count_7000_20000+count_20000)
print("percentage of subscription", pct_of_7000_20000*100)
pct_of_20000 = count_20000/(count_7000+count_7000_20000+count_20000)
print("percentage of subscription", pct_of_20000*100)

data = pd.read_csv('car_price_prediction_expanded.csv',header=0)
data = data.dropna()
#print(data.shape)
print(list(data.columns))

#sns.countplot(x='Price',data=data)
#lt.show()
#plt.savefig('count_plot')

def categorize_price(price):
    if price < 8000:
        return 1
    elif 8000 < price < 18500:
        return 2
    else:
        return 3


def m3():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix

    #data['Manufacturer'] = np.where(data['Manufacturer'] == 'LEXUS', 'Basic', data['Manufacturer'])
    #data['Manufacturer'] = np.where(data['Manufacturer'] == 'HONDA', 'Basic', data['Manufacturer'])
    data_prepared = data.copy()

    data_prepared = data_prepared.drop(columns=["ID", "Model", 'Engine volume', 'Color', 'Doors'])  # Assuming 'Model' is too specific

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


    lr_model = LogisticRegression(max_iter=4000, random_state=42)
    lr_model.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    y_pred = lr_model.predict(X_test)

    # Оценка модели
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr_model.score(X_test, y_test)))
    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix)
    print(classification_report(y_test, y_pred))
    logit_roc_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test), multi_class='ovr', average='macro')

    print(f"ROC-AUC Score: {logit_roc_auc:.2f}")

    # ROC-кривая для многоклассовой классификации
    plt.figure()

    for i in range(len(lr_model.classes_)):
        fpr, tpr, _ = roc_curve(y_test == i + 1, lr_model.predict_proba(X_test)[:, i])  # Прогнозы для каждого класса
        plt.plot(fpr, tpr,
                 label=f"Class {i + 1} (area = {roc_auc_score(y_test == i + 1, lr_model.predict_proba(X_test)[:, i]):.2f})")

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


