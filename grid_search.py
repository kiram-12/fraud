from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
def grid_search(df,model):
    df_x = df.iloc[:,:-1]
    df_y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.3)
    if(model=='svm'):
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        c_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5]
        gammas = [0.1, 1, 10, 100]
        clf = SVC()
        clf.fit(X_train, y_train)
        param_grid = {'kernel': kernels, 'C': c_values, 'gamma': gammas}
        grid = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1)
    elif (model=='knn'):
        neighbors = list(range(1, 21))
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': neighbors}
        grid = GridSearchCV(knn, param_grid, cv=10, n_jobs=-1)
        
    grid.fit(X_train, y_train)
    print("Best parameters for model",model,":", grid.best_params_)
    



    
