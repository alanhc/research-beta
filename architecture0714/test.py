import pandas as pd
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
"""
dataset_100
pic_100
"""

train_dataset = ['fewer_light_100', 'pic_100']

for model_dataset in train_dataset:
    for test_dataset in train_dataset:
        if model_dataset != test_dataset:
            print('=====',model_dataset, test_dataset)

            base = '../../dataset/'+test_dataset+'/'
            dataset = "origin/"
            model_path = '../../dataset/'+model_dataset+'/'+dataset+'model-rf-v6.pkl'
            all_data = pd.read_csv(base+dataset+'data-6.csv')
            X_test = all_data[['iou', 'min', 'std', 'y', 'area']]
            y_test = all_data['answers']

            model = load(model_path)
            print(model[1])
            y_pred = model.predict(X_test)

            print(confusion_matrix(y_pred,y_test))
            print(classification_report(y_pred, y_test))
            y_pred = pd.DataFrame({'predict':y_pred})
            y_pred.to_csv(base+dataset+'data-6-test.csv')
