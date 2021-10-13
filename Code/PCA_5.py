# -*- coding: utf-8 -*-
# @Author: Wenbo Mo
# @Last Modified by:   Wenbo Mo
# @Last Modified date: 2021-08-30

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# data set path defination
data_path_Train = 'D:\\Data\\5-class'

OutputModelName = 'D:\\Model\\PCA_5'

data_length = 895
train_data_ratio = 0.6
train_times = 1

# Training and test datasets ##########################################################################################################
# load data
x_tensor_list_M = [] #x-features, y-label
y_tensor_list_M = []
x_tensor_list_S = []
y_tensor_list_S = []
x_tensor_list_SC = []
y_tensor_list_SC = []
x_tensor_list_H = []
y_tensor_list_H = []
x_tensor_list_O = []
y_tensor_list_O = []

file_count = 0
MSample = 0
SSample = 0
SCSample = 0
HSample = 0
OSample = 0
# load all file in this path
for foldername in os.listdir(data_path_Train):
	for filename in os.listdir(data_path_Train+'\\'+foldername):
		file_count = file_count+1
		# load data
		wavenumber = np.loadtxt(data_path_Train+'\\'+foldername+'\\'+filename)[0]
		data = np.loadtxt(data_path_Train+'\\'+foldername+'\\'+filename)[1:]
		for i,line in enumerate(data):
			line = (line-min(line))/(max(line)-min(line))
			# add data to train list
			if foldername == 'MERS':
				x_tensor_list_M.append(line)
				MSample = MSample + 1
				y_tensor_list_M.append([1])
			elif foldername == 'SARS':
				x_tensor_list_S.append(line)
				SSample = SSample + 1
				y_tensor_list_S.append([2])
			elif foldername == 'SARS-Cov-2':
				x_tensor_list_SC.append(line)
				SCSample = SCSample + 1
				y_tensor_list_SC.append([3])
			elif foldername == 'HKU1':
				x_tensor_list_H.append(line)
				HSample = HSample + 1
				y_tensor_list_H.append([4])
			elif foldername == 'OC43':
				x_tensor_list_O.append(line)
				OSample = OSample + 1
				y_tensor_list_O.append([5])

x_tensor_list_M = np.asarray(x_tensor_list_M)
y_tensor_list_M = np.asarray(y_tensor_list_M)
x_tensor_list_S = np.asarray(x_tensor_list_S)
y_tensor_list_S = np.asarray(y_tensor_list_S)
x_tensor_list_SC = np.asarray(x_tensor_list_SC)
y_tensor_list_SC = np.asarray(y_tensor_list_SC)
x_tensor_list_H = np.asarray(x_tensor_list_H)
y_tensor_list_H = np.asarray(y_tensor_list_H)
x_tensor_list_O = np.asarray(x_tensor_list_O)
y_tensor_list_O = np.asarray(y_tensor_list_O)

print('MERS sample in data : ',str(MSample))
print('SARS sample in data : ',str(SSample))
print('SARS-Cov-2 sample in data : ',str(SCSample))
print('HKU1 sample in data : ',str(HSample))
print('OC43 sample in data : ',str(OSample))

per = np.random.permutation(x_tensor_list_M.shape[0])
x_tensor_list_M = x_tensor_list_M[per,:]
x_train_tensor_list_M = np.asarray(x_tensor_list_M[0:int(len(x_tensor_list_M)*train_data_ratio)])
x_test_tensor_list_M = np.asarray(x_tensor_list_M[int(len(x_tensor_list_M)*train_data_ratio):])
y_train_tensor_list_M = np.asarray(y_tensor_list_M[0:int(len(y_tensor_list_M)*train_data_ratio)])
y_test_tensor_list_M = np.asarray(y_tensor_list_M[int(len(y_tensor_list_M)*train_data_ratio):])

per = np.random.permutation(x_tensor_list_S.shape[0])
x_tensor_list_S = x_tensor_list_S[per,:]
x_train_tensor_list_S = np.asarray(x_tensor_list_S[0:int(len(x_tensor_list_S)*train_data_ratio)])
x_test_tensor_list_S = np.asarray(x_tensor_list_S[int(len(x_tensor_list_S)*train_data_ratio):])
y_train_tensor_list_S = np.asarray(y_tensor_list_S[0:int(len(y_tensor_list_S)*train_data_ratio)])
y_test_tensor_list_S = np.asarray(y_tensor_list_S[int(len(y_tensor_list_S)*train_data_ratio):])

per = np.random.permutation(x_tensor_list_SC.shape[0])
x_tensor_list_SC = x_tensor_list_SC[per,:]
x_train_tensor_list_SC = np.asarray(x_tensor_list_SC[0:int(len(x_tensor_list_SC)*train_data_ratio)])
x_test_tensor_list_SC = np.asarray(x_tensor_list_SC[int(len(x_tensor_list_SC)*train_data_ratio):])
y_train_tensor_list_SC = np.asarray(y_tensor_list_SC[0:int(len(y_tensor_list_SC)*train_data_ratio)])
y_test_tensor_list_SC = np.asarray(y_tensor_list_SC[int(len(y_tensor_list_SC)*train_data_ratio):])

per = np.random.permutation(x_tensor_list_H.shape[0])
x_tensor_list_H = x_tensor_list_H[per,:]
x_train_tensor_list_H = np.asarray(x_tensor_list_H[0:int(len(x_tensor_list_H)*train_data_ratio)])
x_test_tensor_list_H = np.asarray(x_tensor_list_H[int(len(x_tensor_list_H)*train_data_ratio):])
y_train_tensor_list_H = np.asarray(y_tensor_list_H[0:int(len(y_tensor_list_H)*train_data_ratio)])
y_test_tensor_list_H = np.asarray(y_tensor_list_H[int(len(y_tensor_list_H)*train_data_ratio):])

per = np.random.permutation(x_tensor_list_O.shape[0])
x_tensor_list_O = x_tensor_list_O[per,:]
x_train_tensor_list_O = np.asarray(x_tensor_list_O[0:int(len(x_tensor_list_O)*train_data_ratio)])
x_test_tensor_list_O = np.asarray(x_tensor_list_O[int(len(x_tensor_list_O)*train_data_ratio):])
y_train_tensor_list_O = np.asarray(y_tensor_list_O[0:int(len(y_tensor_list_O)*train_data_ratio)])
y_test_tensor_list_O = np.asarray(y_tensor_list_O[int(len(y_tensor_list_O)*train_data_ratio):])

x_train_tensor_list = np.vstack((x_train_tensor_list_M,x_train_tensor_list_S,x_train_tensor_list_SC,x_train_tensor_list_H,x_train_tensor_list_O))
y_train_tensor_list = np.vstack((y_train_tensor_list_M,y_train_tensor_list_S,y_train_tensor_list_SC,y_train_tensor_list_H,y_train_tensor_list_O))
per = np.random.permutation(x_train_tensor_list.shape[0])
x_train_tensor_list = x_train_tensor_list[per,:]
y_train_tensor_list = y_train_tensor_list[per,:]

num = 2
pca=PCA(n_components=num)     # load PCA model, set components
# use training data to fit PCA model and transform
x_train_tensor_list_ = pca.fit_transform(x_train_tensor_list)
print(pca.explained_variance_ratio_[0],pca.explained_variance_ratio_[1])
# transform test data
x_test_tensor_list_M=pca.transform(x_test_tensor_list_M)
x_test_tensor_list_S=pca.transform(x_test_tensor_list_S)
x_test_tensor_list_SC=pca.transform(x_test_tensor_list_SC)
x_test_tensor_list_H=pca.transform(x_test_tensor_list_H)
x_test_tensor_list_O=pca.transform(x_test_tensor_list_O)

font = {'family':'Arial','weight':'normal','size':12}
plt.figure(figsize=(8,6),dpi=600)
plt.scatter(x_test_tensor_list_M[:,0],x_test_tensor_list_M[:,1],alpha=0.6,label='MERS-CoV')
plt.scatter(x_test_tensor_list_S[:,0],x_test_tensor_list_S[:,1],alpha=0.6,label='SARS-CoV')
plt.scatter(x_test_tensor_list_SC[:,0],x_test_tensor_list_SC[:,1],alpha=0.6,label='SARS-CoV-2')
plt.scatter(x_test_tensor_list_H[:,0],x_test_tensor_list_H[:,1],alpha=0.6,label='HCoV-HKU1')
plt.scatter(x_test_tensor_list_O[:,0],x_test_tensor_list_O[:,1],alpha=0.6,label='HCoV-OC43')
plt.legend(prop=font)
plt.xlabel('PC1 ('+str(round(pca.explained_variance_ratio_[0]*100,2))+'%)', fontproperties=font)
plt.ylabel('PC2 ('+str(round(pca.explained_variance_ratio_[1]*100,2))+'%)', fontproperties=font)
plt.xticks(fontproperties=font)
plt.yticks(fontproperties=font)
plt.show()

test_acc_SVM = np.zeros((5,train_times))
test_matrix = np.zeros((5,5,train_times))
i = 0
while i < train_times:
	print('Train times:',str(i+1),'/',str(train_times))

	# SVM model
	model0 = Pipeline([ ("scaler", StandardScaler()),
	                                ("svm_clf", SVC())
	                            ])
	model0.fit(x_train_tensor_list_, y_train_tensor_list.reshape((y_train_tensor_list.shape[0])))

	# Test dataset classification accuracy ############################################################################################################
	def Predict(x_test_tensor_list):
		y_predict_tensor_list = model0.predict(x_test_tensor_list)
		j = 0
		P1 = 0
		P2 = 0
		P3 = 0
		P4 = 0
		P5 = 0
		while j < len(y_predict_tensor_list):
			if  y_predict_tensor_list[j] == 1: #记录判断为阳性的点数
				P1 += 1
			elif y_predict_tensor_list[j] == 2:
				P2 += 1
			elif y_predict_tensor_list[j] == 3:
				P3 += 1
			elif y_predict_tensor_list[j] == 4:
				P4 += 1
			elif y_predict_tensor_list[j] == 5:
				P5 += 1
			j = j+1
		return np.array([P1,P2,P3,P4,P5])/len(y_predict_tensor_list)*100

	print(Predict(x_test_tensor_list_M))
	print(Predict(x_test_tensor_list_S))
	print(Predict(x_test_tensor_list_SC))
	print(Predict(x_test_tensor_list_H))
	print(Predict(x_test_tensor_list_O))
	test_matrix[0,:,i] = Predict(x_test_tensor_list_M)
	test_matrix[1,:,i] = Predict(x_test_tensor_list_S)
	test_matrix[2,:,i] = Predict(x_test_tensor_list_SC)
	test_matrix[3,:,i] = Predict(x_test_tensor_list_H)
	test_matrix[4,:,i] = Predict(x_test_tensor_list_O)
	test_acc_SVM[0,i] = Predict(x_test_tensor_list_M)[0]
	test_acc_SVM[1,i] = Predict(x_test_tensor_list_S)[1]
	test_acc_SVM[2,i] = Predict(x_test_tensor_list_SC)[2]
	test_acc_SVM[3,i] = Predict(x_test_tensor_list_H)[3]
	test_acc_SVM[4,i] = Predict(x_test_tensor_list_O)[4]


	i = i+1

print('PCA test acc:', np.mean(test_acc_SVM,axis=1))
print('PCA test acc std:', np.std(test_acc_SVM,axis=1))
print('PCA test matrix:', np.mean(test_matrix,axis=2))

print('PCA test acc:', np.mean(np.mean(test_acc_SVM,axis=1)))
