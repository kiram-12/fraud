
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

df=pd.read_csv('/content/creditcard.csv')
df.dropna(inplace=True)
print(df.head(10))

# Split dataset into X and Y
df_x = df.iloc[:,:-1]
df_y = df.iloc[:,-1]


# Standardize features
#scaler = StandardScaler()
#df_x = scaler.fit_transform(df_x)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)

print(f"X_train shape {X_train.shape}")
print(f"Y_train shape {y_train.shape}")
print(f"X_test shape {X_test.shape}")
print(f"Y_test shape {y_test.shape}")

y_test = pd.Series(y_test)

import matplotlib.pyplot as plt

colors = {0: 'red', 1: 'blue'}
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 2], c=y_test, cmap=plt.cm.coolwarm)
plt.show()

"""# Implement Random search to find best value of k"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
neighbors = list(range(1, 21))

knn = KNeighborsClassifier()

param_grid = {'n_neighbors': neighbors}

grid = GridSearchCV(knn, param_grid, cv=10, n_jobs=-1)

grid.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid.best_params_
print("Best hyperparameters:", best_params)

neighbors = list(range(3, 21))
neighbors

"""# Train KNN with best value of K"""

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 1)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n", classification_report(y_pred, y_test))
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
np.set_printoptions(precision=2)
plt.figure()

model = KNeighborsClassifier(n_neighbors = 4)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n", classification_report(y_pred, y_test))
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
np.set_printoptions(precision=2)
plt.figure()

model = KNeighborsClassifier(n_neighbors = 1)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n", classification_report(y_pred, y_test))
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
np.set_printoptions(precision=2)
plt.figure()

"""Implement function to plot decision boundry"""

kf = KFold(n_splits=10, shuffle=True)

acc_arr = np.empty((10, 1))
f1_arr = np.empty((10, 1))
cnf_arr= []
x = 0
for train_index, test_index in kf.split(df_x, df_y):
    X_train, X_test = df_x.iloc[train_index], df_x.iloc[test_index]
    y_train, y_test = df_y.iloc[train_index], df_y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
    print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
    print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
    print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
    print("\n", classification_report(y_pred, y_test))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    acc_arr[x] = accuracy_score(y_test, y_pred)
    f1_arr[x] = f1_score(y_test, y_pred)

    x = x+ 1

print("%0.2f f1 score with a standard deviation of %0.2f" %
      (f1_arr.mean(), f1_arr.std()))
print("%0.2f accuracy with a standard deviation of %0.2f" %
      (acc_arr.mean(), acc_arr.std()))