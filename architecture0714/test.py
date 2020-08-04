import pandas as pd
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


base = '../../dataset/dataset_100/'
all_data = pd.read_csv(base+'data.csv')
X_test = all_data[['iou', 'min', 'std', 'y', 'area']]
y_test = all_data['answers']

model = load('models/'+'model.pkl')

y_pred = model.predict(X_test)
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred, y_test))
