from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss
import numpy as np 
import matplotlib.pyplot as plt 

def svm_model(df, ker, gamma, c, tol, undersample_method):
    df_x = df.iloc[:,:-1]
    df_y = df.iloc[:,-1]
    
    # Undersampling
    X_resampled, y_resampled = under_sampling(df, undersample_method)
    
    # Splitting data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)
    
    # SVM model
    model = SVC(kernel=ker, gamma=gamma, C=c, tol=tol, verbose=1)
    cv_scores = cross_val_score(model, X_train, y_train, cv=10)
    print('Cross-validation scores:', cv_scores)
    print('Average cross-validation score:', np.mean(cv_scores))
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
    print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
    print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
    print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
    print("\n", classification_report(y_pred, y_test))
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    np.set_printoptions(precision=2)
    plt.figure()

def under_sampling(df, method):
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    if method == 'random':
        print("Before sampling:", Y.value_counts()) 
        rus = RandomUnderSampler(sampling_strategy='auto', random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X.values, Y)
        print("After sampling:", y_resampled.value_counts())
        plt.histplot(y_resampled)
        plt.show()
    elif method == 'tomeklinks':
        print("Before sampling:", Y.value_counts()) 
        tl = TomekLinks(sampling_strategy='auto')
        X_resampled, y_resampled = tl.fit_resample(X, Y)
        print("After sampling:", y_resampled.value_counts())
        plt.histplot(y_resampled)
        plt.show()
    elif method == 'enn':
        print("Before sampling:", Y.value_counts()) 
        enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=31, kind_sel='all')
        X_resampled, y_resampled = enn.fit_resample(X, Y)
        print("After sampling:", y_resampled.value_counts())
        plt.histplot(y_resampled)
        plt.show()
    elif method == 'nearmiss1':
        print("Before sampling:", Y.value_counts()) 
        nm1 = NearMiss(
            sampling_strategy='auto',  # undersamples only the majority class
            version=1,
            n_neighbors=3,
            n_jobs=4)
        X_resampled, y_resampled = nm1.fit_resample(X, Y)
        print("After sampling:", y_resampled.value_counts())
        plt.histplot(y_resampled)
        plt.show()
    elif method == 'nearmiss2':
        print("Before sampling:", Y.value_counts()) 
        nm1 = NearMiss(
            sampling_strategy='auto',
            version=2,
            n_neighbors=3,
            n_jobs=4)
        X_resampled, y_resampled = nm1.fit_resample(X, Y)
        print("After sampling:", y_resampled.value_counts())
        plt.histplot(y_resampled)
        plt.show()
    return X_resampled, y_resampled
