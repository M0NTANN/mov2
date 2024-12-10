import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('AB_NYC_2019.csv')
#data = data[data['neighbourhood_group'] == 'Brooklyn' or data['neighbourhood_group'] == 'Manhattan' ]
data['neighbourhood_group']=np.where(data['neighbourhood_group'] =='Brooklyn', 'Basic', data['neighbourhood_group'])
data['neighbourhood_group']=np.where(data['neighbourhood_group'] =='Manhattan', 'Basic', data['neighbourhood_group'])


count_no_sub = len(data[data['room_type']=='Private room'])
count_sub = len(data[data['room_type']=='Entire home/apt'])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Убираем ненужные столбцы
columns_to_drop = ['id', 'name', 'host_id', 'host_name', 'last_review', 'neighbourhood']
data_cleaned = data.drop(columns=columns_to_drop)

# Заполнение пропусков
data_cleaned['reviews_per_month'].fillna(0, inplace=True)


# Кодирование целевой переменной (neighbourhood_group)
label_encoder = LabelEncoder()
data_cleaned['neighbourhood_group'] = label_encoder.fit_transform(data_cleaned['neighbourhood_group'])

# Кодирование категориальных признаков (например, room_type)
data_encoded = pd.get_dummies(data_cleaned, columns=['room_type'], drop_first=True)

# Разделение данных на признаки и целевую переменную
X = data_encoded.drop(columns=['neighbourhood_group'])
y = data_encoded['neighbourhood_group']


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
fpr, tpr, thresholds = roc_curve(y_test, lr_model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()




