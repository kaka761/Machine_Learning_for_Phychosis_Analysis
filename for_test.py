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
from sklearn.metrics import classification_report,plot_confusion_matrix
import seaborn as sns



random_state = 2021

test = pd.read_csv('./Projects/MRI/ML/Fea_test/FEP_HC/ARMS_T_FD25_30_2.csv')
test = test.sample(frac=1.0, random_state=random_state)


from sklearn.preprocessing import StandardScaler

x = test
#standardscaler = StandardScaler()
#x = standardscaler.fit_transform(x)
#print(x)

from joblib import dump, load
from sklearn.svm import SVC
model = load('./Projects/MRI/ML/results/models/MLP_FEP_HC.joblib')
pred = model.predict(x)
print(pred)