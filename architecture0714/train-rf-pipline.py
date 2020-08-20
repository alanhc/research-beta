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
save_path = base+dataset+'model-rf-v2.pkl'
all_data = pd.read_csv(base+dataset+'data-2.csv')
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



steps = [('scaler', StandardScaler()), ('rf', RandomForestClassifier())]
pipeline = Pipeline(steps) 

parameteres = {'rf__n_estimators': n_estimators,
               'rf__max_features': max_features,
               'rf__max_depth': max_depth,
               'rf__min_samples_split': min_samples_split,
               'rf__min_samples_leaf': min_samples_leaf,
               'rf__bootstrap': bootstrap}
rf_random = GridSearchCV(pipeline, param_grid=parameteres, n_jobs=-1)


rf_random.fit(X_train, y_train)

print(rf_random.best_params_)

dump(rf_random.best_estimator_, save_path)
model = load(save_path)

y_pred = model.predict(X_train)

print(confusion_matrix(y_pred,y_train))
print(classification_report(y_pred, y_train))

#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74