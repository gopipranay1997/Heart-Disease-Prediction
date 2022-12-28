import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv("/home/k8user/Pictures/vae-dr/archive (1)/data.csv")
df1 = df.drop(['Unnamed: 32', 'id', 'diagnosis'], axis=1)
X_t = df1
y_t = df['diagnosis']
y_t = np.where(y_t=='B',0,1)

X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.1, random_state=42, stratify=y_t)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

lr = LogisticRegression()
lr_model = lr.fit(X_train_scaled, y_train)
train_pred = lr_model.predict(X_train_scaled)
test_pred = lr_model.predict(X_test_scaled)

AUC_Train_Score = roc_auc_score(y_train, train_pred)
AUC_Test_Score = roc_auc_score(y_test, test_pred)

with open("metrics.txt", "w") as outfile:
    outfile.write("Training AUC_SCORE: %2.1f%%\n" % AUC_Train_Score)
    outfile.write("Testing AUC_SCORE: %2.1f%%\n" % AUC_Test_Score)
