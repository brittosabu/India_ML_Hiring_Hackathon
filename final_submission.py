import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.drop(['loan_id','source','financial_institution','origination_date','first_payment_date','insurance_type','loan_purpose'],axis=1,inplace=True)
test.drop(['loan_id','source','financial_institution','origination_date','first_payment_date','insurance_type','loan_purpose'],axis=1,inplace=True)

pos= len(train.columns)-1

train.insert(pos,'deliquent_sum',train.iloc[:,pos-12:pos].sum(axis=1).astype('int'))

test.insert(pos,'deliquent_sum',test.iloc[:,pos-12:pos].sum(axis=1).astype('int'))

drop1=train[train['unpaid_principal_bal']>610000].index
train.drop(drop1,inplace=True)

drop2=train[(train['interest_rate']>6) | (train['interest_rate']<2.7)].index
train.drop(drop2,inplace=True)

drop3=train[train['loan_to_value']<19].index
train.drop(drop3,inplace=True)

drop4 = train[train['debt_to_income_ratio']>60].index
train.drop(drop4,inplace=True)

drop5 = train[train['borrower_credit_score']<590].index
train.drop(drop5,inplace=True)

drop6 = train[train['deliquent_sum']>53].index
train.drop(drop6,inplace=True)

train.drop(['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10'],axis=1,inplace=True)
test.drop(['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10'],axis=1,inplace=True)
train.drop('co-borrower_credit_score',axis=1,inplace=True)
test.drop('co-borrower_credit_score',axis=1,inplace=True)

y=np.asarray(train.m13)
X=train.drop('m13',axis=1)


#total_data = pd.concat([train.iloc[:,:11],test],ignore_index=True)
#total_scaled_data=scalar.fit_transform(total_data)
#trainX = total_scaled_data[:114774]
#testX = total_scaled_data[114774:]


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.3,random_state=10)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score

rf = RandomForestClassifier(random_state=7)
rf1 = RandomForestClassifier(n_estimators=5000,max_depth=35,min_samples_split=2,min_samples_leaf=2)
rf1.fit(X_train_res, y_train_res.ravel())
y_pred = rf1.predict(X_test)
print("{} with F1: {}".format(accuracy_score(y_test,y_pred),f1_score(y_test,y_pred)))

y_predict = rf1.predict(test)
predictions = pd.DataFrame(y_predict,columns=['m13']).astype('int')
predictions.insert(0,'loan_id',np.arange(1,35867))

print(predictions.m13.value_counts())

predictions.to_csv('rf_5000_35_2_2_noscale_predicted.csv',index=None,header=True)

#y_predict = rf1.predict(testX)

#predictions = pd.DataFrame(y_predict,columns=['m13']).astype('int')
#predictions.insert(0,'loan_id',np.arange(1,35867))

#print(predictions.m13.value_counts())

#predictions.to_csv('rf_1500_22_2_2_predicted.csv',index=None,header=True)
