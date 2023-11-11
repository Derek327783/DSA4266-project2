import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import xgboost as xgb

train_data =  pd.read_csv("./data/train_OG.csv")
test_data =  pd.read_csv("./data/test_OG.csv")
train_data = train_data.drop(columns = ['Unnamed: 0'])
test_data = test_data.drop(columns = ['Unnamed: 0'])

def random_undersample(train_data):
    rus = RandomUnderSampler()
    X = train_data.drop(columns = ["label","gene_id","Transcript_ID","Base_seq"])
    Y = train_data['label']
    X_resampled, y_resampled = rus.fit_resample(X, Y)
    output = pd.concat([X_resampled,y_resampled],axis = 1)
    return output

new_train_data = random_undersample(train_data)
X_train = new_train_data.drop('label', axis=1)
Y_train = new_train_data['label']
X_test = test_data.drop(columns = ["label","gene_id","Transcript_ID","Base_seq"],axis = 1)
Y_test = test_data['label']


model = xgb.XGBClassifier(
    objective='binary:logistic', 
    n_estimators=100,            
    max_depth=3,                
    learning_rate=0.1,          
    subsample=0.8,             
    colsample_bytree=0.8,       
)

model.fit(X_train, Y_train)

Y_pred = model.predict_proba(X_test)[:, 1]


fpr, tpr, _ = roc_curve(Y_test, Y_pred)
roc_auc = auc(fpr, tpr)
#accuracy = accuracy_score(Y_test, Y_pred)
#print(f'Accuracy: {accuracy * 100:.2f}%')
print(roc_auc)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('plot_xg.png')
