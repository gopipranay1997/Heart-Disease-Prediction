import pandas as pd
import numpy as np


df4 = pd.read_csv("leukemia_big.csv")
df4_1 = (df4.T).reset_index()
df4_1.index = df4_1.index.astype(str)
df4_labels = pd.DataFrame(df4_1['index'].str.split('.',1).tolist(),columns = ['label','remove'])
df4_labels.drop('remove', inplace=True, axis=1)
X4 = df4_1.drop('index',axis=1)
y4 = df4_labels
y4['label'] = np.where(y4['label']=='ALL',0,1)
from sklearn.model_selection import train_test_split
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.1, random_state=42, stratify=y4)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train4, y_train4)
y_pred_test = dt.predict(X_test4)
y_pred_train = dt.predict(X_train4)
from sklearn.metrics import accuracy_score, roc_auc_score
# print(accuracy_score(y_test4, y_pred))
AUC_Test_Score = roc_auc_score(y_test4, y_pred_test)
AUC_Train_Score = roc_auc_score(y_train4, y_pred_train)

with open("metrics.txt", "w") as outfile:
  outfile.write("Training AUC_SCORE: %2.1f%%\n" % AUC_Train_Score)
  outfile.write("Testing AUC_SCORE: %2.1f%%\n" % AUC_Test_Score)
