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
base = '../../dataset/dataset_100/'
dataset = "origin/"
train_dataset = ['dataset_100', 'pic_100']

for dataset_name in train_dataset:
    base = '../../dataset/'+dataset_name+'/'
    save_path = base+dataset+'model-svm-v5.pkl'
        
    all_data = pd.read_csv(base+dataset+'data-3.csv')
    print('all_data:',all_data.shape)

    all_count = all_data.answers.value_counts().sort_index()
    max_count = all_count.max()
    len_data = len(all_count) 
    print(all_count)

    df_class=[]
    for i in range(len_data):
        idx=all_count.index[i]
        print(i)
        df_class.append(all_data[all_data['answers'] == idx])

    df_test_over = pd.DataFrame()
    #print('===',all_count[2])
    for i in range(len_data):
        
        idx=all_count.index[i]
        if all_count[idx]!= max_count:
            df_class_over = df_class[i].sample(max_count, replace=True)
            df_test_over = pd.concat([df_test_over,df_class_over], axis=0)
        else:
            df_test_over = pd.concat([df_test_over,df_class[i]], axis=0)

    print(df_test_over.answers.value_counts())

    data = df_test_over
    X_train = data[['iou', 'min', 'std', 'y', 'area']]
    y_train = data['answers']



    steps = [('scaler', StandardScaler()), ('SVM', SVC())]
    pipeline = Pipeline(steps) 

    parameteres = {'SVM__kernel':['linear','poly', 'rbf', 'sigmoid'],'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}
    parameteres2 = {'SVM__kernel':['rbf'],'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}
    parameteres3 = {'SVM__kernel':['rbf'],'SVM__C':[0.001,1,100,10e5], 'SVM__gamma':[0.1,0.01]}
    parameteres4 = {}


    svm = GridSearchCV(pipeline, param_grid=parameteres4, n_jobs=-1)


    svm.fit(X_train, y_train)

    print(svm.best_params_)

    dump(svm.best_estimator_, save_path)
    model = load(save_path)

    y_pred = model.predict(X_train)

    print(confusion_matrix(y_pred,y_train))
    print(classification_report(y_pred, y_train))

    # save
    y_pred = pd.DataFrame({'predict':y_pred})
    y_pred.to_csv(base+dataset+'data-3-test.csv')

    #https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74