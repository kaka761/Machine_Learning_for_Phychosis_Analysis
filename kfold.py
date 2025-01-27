import sys 
import pandas as pd
import os
import random
import shutil
import sklearn 
import scipy
import numpy as np
import radiomics  #这个库专门用来提取特征
from radiomics import featureextractor
import SimpleITK as sitk  #读取nii文件
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, plot_confusion_matrix
import seaborn as sns



random_state = 2021
###### feature choosing ######
### T-test ###
from scipy.stats import levene, ttest_ind

FEP_data = pd.read_csv('./Projects/MRI/ML/Fea_FD/FD25_30/ARMS_NT.csv') # 1
C_data = pd.read_csv('./Projects/MRI/ML/Fea_FD/FD25_30/FEP.csv') # 0

FEP_data.insert(0,'label', 1) # insert label
C_data.insert(0,'label', 0)

FEP_data = FEP_data.sample(frac=1.0, random_state=random_state)  # shuffle
C_data = C_data.sample(frac=1.0, random_state=random_state)

# delete string feature
cols=[x for i, x in enumerate(FEP_data.columns) if type(FEP_data.iat[1,i]) == str]
FEP_data_train=FEP_data.drop(cols,axis=1)
cols=[x for i, x in enumerate(C_data.columns) if type(C_data.iat[1,i]) == str]
C_data_train=C_data.drop(cols,axis=1)


counts = 0
columns_index =[]
for column_name in FEP_data_train.columns[1:]:
    # levene test: p>0.05, homogeneity of variance;
    if levene(FEP_data_train[column_name], C_data_train[column_name])[1] > 0.05: 
        if ttest_ind(FEP_data_train[column_name],C_data_train[column_name],equal_var=True)[1] < 0.05:
            columns_index.append(column_name)
    else:
        if ttest_ind(FEP_data_train[column_name],C_data_train[column_name],equal_var=False)[1] < 0.05:
            columns_index.append(column_name)

print("No. features after T-test selector:{}".format(len(columns_index)))

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

### Lasso ###
if  not 'label' in columns_index:
    columns_index = ['label'] + columns_index
FEP_train = FEP_data_train[columns_index]  
C_train = C_data_train[columns_index]

data = pd.concat([FEP_train, C_train])
data = data.sample(frac=1.0, random_state=random_state)  # shuffle

# separate the features and label
x = data[data.columns[1:]]
y = data['label']
columnNames = x.columns

lassoCV_x = x.astype(np.float32)
lassoCV_y = y

standardscaler = StandardScaler()
lassoCV_x = standardscaler.fit_transform(lassoCV_x)
lassoCV_x = pd.DataFrame(lassoCV_x,columns=columnNames)

# 5**(-3) < alpha <  5**(-2)
alpha_range = np.logspace(-3,-2, 50, base=5)
#cv=5 means 5 splits for cross-validation
lassoCV_model = LassoCV(alphas = alpha_range, cv=5, max_iter=1000)
#training
lassoCV_model.fit(lassoCV_x,lassoCV_y)


print(lassoCV_model.alpha_)
coef = pd.Series(lassoCV_model.coef_, index=columnNames)
print("the original {} features, remain {} features".format(len(columnNames),sum(coef !=0)))
print("Those features are:")
print(coef[coef !=0])
index = coef[coef !=0].index
lassoCV_x = lassoCV_x[index]


# plot feature correlation coefficient heat map
import seaborn as sns
f, ax= plt.subplots(figsize = (10, 10))
sns.heatmap(lassoCV_x.corr(),annot=True,cmap='coolwarm',annot_kws={'size':10,'weight':'bold', },ax=ax) # plot confusion matrix
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,va="top",ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
plt.show()

# plot feature coefficient histogram
weight = coef[coef !=0].to_dict()
# sort the values
weight = dict(sorted(weight.items(),key=lambda x:x[1],reverse=False))
plt.figure(figsize=(8,6))
plt.title('characters classification weight',fontsize=15)
plt.xlabel(u'weighted value',fontsize=14)
plt.ylabel(u'feature')
plt.barh(range(len(weight.values())), list(weight.values()),tick_label = list(weight.keys()),alpha=0.6, facecolor = 'blue', edgecolor = 'black', label='feature weight')
plt.legend(loc=4)
plt.show()

# plot error bars
MSEs = lassoCV_model.mse_path_
mse = list()
std = list()
for m in MSEs:
    mse.append(np.mean(m))
    std.append(np.std(m))

plt.figure(figsize=(8,6))
plt.errorbar(lassoCV_model.alphas_, mse, std,fmt='o:',ecolor='lightblue',
			elinewidth=3,ms=5,mfc='wheat',mec='salmon',capsize=3)
plt.axvline(lassoCV_model.alpha_, color='red', ls='--')
plt.title('Errorbar')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.show()


x = data[data.columns[1:]]
y = data['label']
columnNames = x.columns
lassoCV_x = x.astype(np.float32)
lassoCV_y = y
lassoCV_x = standardscaler.transform(lassoCV_x)
lassoCV_x = pd.DataFrame(lassoCV_x,columns=columnNames)
coefs = lassoCV_model.path(lassoCV_x,lassoCV_y, alphas=alpha_range, max_iter=1000)[1].T
plt.plot(lassoCV_model.alphas,coefs,'-')
plt.axvline(lassoCV_model.alpha_, color='red', ls='--')
plt.xlabel('Lambda')
plt.ylabel('coef')
plt.show()

index_ = coef[coef != 0].index
x = x[index_] 
y = y.values
print(x.shape)
print(y.shape)
standardscaler = StandardScaler()
x = standardscaler.fit_transform(x)

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, multilabel_confusion_matrix
import torch
from torchmetrics import Specificity, Recall, Accuracy
from joblib import dump, load


model1 = RandomForestClassifier(n_estimators=30, random_state=random_state)    
model2 = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)
model3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=4, random_state=0)
model4 = AdaBoostClassifier(n_estimators= 100)
model5 = LogisticRegression(penalty='l2', C=0.5, solver='liblinear')
model6 = SGDClassifier(max_iter=10000, tol=1e-4)
model7 = KNeighborsClassifier(n_neighbors=5)
model8 = MLPClassifier(hidden_layer_sizes=(10,),alpha=0.01,max_iter=10000)
model9 = DecisionTreeClassifier(criterion='entropy', splitter='best')
model10 = GaussianNB()
model11 = SVC(kernel='rbf', C=5, gamma=0.05, probability=True)
kfold = StratifiedKFold(n_splits=10) #, shuffle=True, random_state=random_state)
y = torch.Tensor(y)
spe = Specificity(task= 'binary', num_classes = 2)
sen= Recall(task= 'binary',num_classes = 2)
#acc = Accuracy(task = 'binary', num_classes = 2)
acc = (spe + sen)/2
s1 = cross_val_predict(model1, x, y, cv=kfold)
s1 = torch.Tensor(s1)
print('RF_:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s1, y), sen(s1, y), acc(s1, y)))

s3 = cross_val_predict(model3, x, y, cv=kfold)
s3 = torch.Tensor(s3)
print('GB_:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s3, y), sen(s3, y), acc(s3, y)))

s4 = cross_val_predict(model4, x, y, cv=kfold)
s4 = torch.Tensor(s4)
print('AB_:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s4, y), sen(s4, y), acc(s4, y)))

s5 = cross_val_predict(model5, x, y, cv=kfold)
s5 = torch.Tensor(s5)
print('LR_:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s5, y), sen(s5, y), acc(s5, y)))

s6 = cross_val_predict(model6, x, y, cv=kfold)
s6 = torch.Tensor(s6)
print('SGD:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s6, y), sen(s6, y), acc(s6, y)))

s7 = cross_val_predict(model7, x, y, cv=kfold)
s7 = torch.Tensor(s7)
print('KN_:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s7, y), sen(s7, y), acc(s7, y)))

s8 = cross_val_predict(model8, x, y, cv=kfold)
s8 = torch.Tensor(s8)
print('MLP:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s8, y), sen(s8, y), acc(s8, y)))

s9 = cross_val_predict(model9, x, y, cv=kfold)
s9 = torch.Tensor(s9)
print('DT_:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s9, y), sen(s9, y), acc(s9, y)))

s10 = cross_val_predict(model10, x, y, cv=kfold)
s10 = torch.Tensor(s10)
print('GNB:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s10, y), sen(s10, y), acc(s10, y)))

s11 = cross_val_predict(model11, x, y, cv=kfold)
s11 = torch.Tensor(s11)

print('SVC:: spe: {:.4f}'' sen: {:.4f}'' acc: {:.4f}'.format(spe(s11, y), sen(s11, y), acc(s11, y)))


model8.fit(x,y)
dump(model8, './Projects/MRI/ML/results/models/MLP_FEP_HC.joblib')