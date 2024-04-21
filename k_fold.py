
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix,accuracy_score
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import numpy as np 
import Knn
def kfold(k,df,mod):
    df_x=df.iloc[:,:-1]
    df_y=df.iloc[:,-1]
    kf = KFold(n_splits=k, shuffle=True)
    acc_arr = np.empty((10, 1))
    f1_arr = np.empty((10, 1))
    cnf_arr= [] 
    x = 0
    for train_index, test_index in kf.split(df_x, df_y):
        X_train, X_test = df_x[train_index], df_x[test_index]
        y_train, y_test = df_y[train_index], df_y[test_index]
        if(mod=='svm'):
            Knn(df,'rbf',0.1, 1.0, 1e-5,'svm',0)
          
        Knn.fit(X_train, y_train)
        y_pred = Knn.predict(X_test)
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