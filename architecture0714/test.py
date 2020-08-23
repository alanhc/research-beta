import pandas as pd
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
"""
dataset_100
pic_100
"""

base = '../../dataset/dataset_100/'
dataset = "origin/"
model_path = '../../dataset/pic_100/'+dataset+'model-svm-v5.pkl'
all_data = pd.read_csv(base+dataset+'data-3.csv')
X_test = all_data[['iou', 'min', 'std', 'y', 'area']]
y_test = all_data['answers']

model = load(model_path)
print(model[1])
y_pred = model.predict(X_test)

print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred, y_test))
y_pred = pd.DataFrame({'predict':y_pred})
y_pred.to_csv(base+dataset+'data-3-test.csv')
