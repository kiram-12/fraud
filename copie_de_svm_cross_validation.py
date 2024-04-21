
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

df = pd.read_csv("/content/creditcard.csv")
df.dropna(inplace=True)
print(df.head(10))

corr = df.corr()
ax, fig = plt.subplots(figsize=(15, 15))
sns.heatmap(corr, vmin=-1, cmap=plt.cm.Blues, annot=True)
plt.show()



corr[abs(corr['Class']) < 0.3]['Class']

# Split dataset into X and Y
df_x = df.iloc[:,:-1]
df_y = df.iloc[:,-1]

# Standardize features
scaler = StandardScaler()
df_x = scaler.fit_transform(df_x)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)

print(f"X_train shape {X_train.shape}")
print(f"Y_train shape {y_train.shape}")
print(f"X_test shape {X_test.shape}")
print(f"Y_test shape {y_test.shape}")

model = SVC(kernel = 'rbf',verbose=1, random_state = 0).fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n", classification_report(y_pred, y_test))
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
np.set_printoptions(precision=2)
plt.figure()

#Large Value of parameter C => small margin
#Small Value of paramerter C => Large margin
#Gamma high means more curvature.
#Gamma low means less curvature.

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

kernels = ['linear', 'rbf', 'poly', 'sigmoid']
c_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5]
gammas = [0.1, 1, 10, 100]
clf = SVC()
clf.fit(X_train, y_train)
param_grid = {'kernel': kernels, 'C': c_values, 'gamma': gammas}
grid = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)

model_ = SVC(kernel='rbf',gamma=0.1, C=1.0, tol=1e-5, verbose=1).fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n", classification_report(y_pred, y_test))

cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
sns.heatmap((cnf_matrix / np.sum(cnf_matrix)*100),
            annot=True, fmt=".2f", cmap="Blues")

model_ = SVC(kernel='linear',gamma=0.1, C=1e-05, tol=1e-5, verbose=1).fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n", classification_report(y_pred, y_test))

cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
sns.heatmap((cnf_matrix / np.sum(cnf_matrix)*100),
            annot=True, fmt=".2f", cmap="Blues")



"""While doing tests, the different sets of train/test where giving different results
on the F1 score and accuraccy. Thus and to avoid overfitting cross-validation is used in
this experiment to avoid previouse mentioned problems. Cross-validation splits a dataset
into k parts, where

See: https://machinelearningmastery.com/k-fold-cross-validation/
"""

kf = KFold(n_splits=10, shuffle=True)

acc_arr = np.empty((10, 1))
f1_arr = np.empty((10, 1))
cnf_arr= []
x = 0
for train_index, test_index in kf.split(df_x, df_y):
    X_train, X_test = df_x[train_index], df_x[test_index]
    y_train, y_test = df_y[train_index], df_y[test_index]
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

