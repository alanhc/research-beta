import pandas as pd
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#train_dataset = ['dataset_100', 'pic_100', 'fewer_light_100']
train_dataset = ['dataset_100', 'pic_100']

#dataset = "origin/"
dataset = "origin-small/"

#train_data = pd.read_csv()


for model_dataset in train_dataset:
    for test_dataset in train_dataset:
        if model_dataset != test_dataset:

            model_base = '../../dataset/'+model_dataset+'/'
            test_base = '../../dataset/'+test_dataset+'/'
            
            
            test_data = pd.read_csv(test_base+dataset+'data-7-train-'+dataset.split('/')[0]+'.csv')
            X_test = test_data[['iou', 'min', 'std', 'y', 'area']]
            y_test = test_data['answers']
            

            for model_name in ['svm', 'rf']:
                model_path = model_base+dataset+'data-7-train-'+dataset.split('/')[0]+'-model-'+model_name+'.pkl'
                model = load(model_path)
                print(model_path,test_base+dataset+'data-7-train-'+dataset.split('/')[0]+'.csv' )

                print(model[1])
                y_pred = model.predict(X_test)

                print(confusion_matrix(y_pred,y_test))
                print(classification_report(y_pred, y_test))
                #y_pred = pd.DataFrame({'predict':y_pred})
                #y_pred.to_csv(base+dataset+'data-6-test.csv')