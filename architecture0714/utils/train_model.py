
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def rf(X_train=None, y_train=None, save_path=None ):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        parameteres = {'rf__n_estimators': n_estimators,
                'rf__max_features': max_features,
                'rf__max_depth': max_depth,
                'rf__min_samples_split': min_samples_split,
                'rf__min_samples_leaf': min_samples_leaf,
                'rf__bootstrap': bootstrap}


        n_estimators2 = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 4)]
        max_depth2 = [int(x) for x in np.linspace(10, 110, num = 5)]
        parameteres2 = {'rf__n_estimators': n_estimators2,
                    'rf__max_features': max_features,
                    'rf__max_depth': max_depth2,
                    'rf__min_samples_split': min_samples_split,
                    'rf__min_samples_leaf': min_samples_leaf,
                    'rf__bootstrap': bootstrap}
        steps = [('scaler', StandardScaler()), ('rf', RandomForestClassifier())]
        pipeline = Pipeline(steps) 

        parameteres3 = {
            'rf__n_estimators': [50,100,200],
            'rf__max_features': ['auto', 'sqrt', 'log2'],
            'rf__max_depth' : [4,8],
            'rf__criterion' :['gini', 'entropy']
        }
        #print(parameteres3)
        rf_random = GridSearchCV(pipeline, param_grid=parameteres3,cv=5, n_jobs=-1)


        rf_random.fit(X_train, y_train)
        print(rf_random.best_params_)
        
        dump(rf_random.best_estimator_, save_path)
        model = load(save_path)

        y_pred = model.predict(X_train)

        print(confusion_matrix(y_pred,y_train))
        print(classification_report(y_pred, y_train))

def svm(X_train=None, y_train=None, save_path=None ):
    steps = [('scaler', StandardScaler()), ('SVM', SVC())]
    pipeline = Pipeline(steps) 

    parameteres = {'SVM__kernel':['linear','poly', 'rbf', 'sigmoid'],'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}
    parameteres2 = {'SVM__kernel':['rbf'],'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}
    parameteres3 = {'SVM__kernel':['rbf'],'SVM__C':[0.001,1,100,10e5], 'SVM__gamma':[0.1,0.01]}
    parameteres4 = {}


    svm = GridSearchCV(pipeline, param_grid=parameteres, n_jobs=-1)


    svm.fit(X_train, y_train)

    print(svm.best_params_)

    dump(svm.best_estimator_, save_path)
    model = load(save_path)

    y_pred = model.predict(X_train)

    print(confusion_matrix(y_pred,y_train))
    print(classification_report(y_pred, y_train))