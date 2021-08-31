# -*- coding: utf-8 -*-
# @Author: Wenbo Mo
# @Last Modified by:   Wenbo Mo
# @Last Modified date: 2021-08-30

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import os
import csv
import sys
import random
import math

# data set path defination
data_path_Train = 'D:\\Data\\5-class'

InputModelName = 'D:\\Model\\MLP_3'
OutputModelName1 = 'D:\\Model\\MLP_5'

data_length = 895
train_data_ratio = 0.8
train_times = 10

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
	for filename in os.listdir(data_path_Train+'\\'+foldername+'\\Txt'):
		file_count = file_count+1
		# print('load train file:',data_path_Train+'\\'+foldername+'\\Txt\\'+filename,file_count)
		# load data
		wavenumber = np.loadtxt(data_path_Train+'\\'+foldername+'\\Txt\\'+filename)[0]
		data = np.loadtxt(data_path_Train+'\\'+foldername+'\\Txt\\'+filename)[1:]
		for i,line in enumerate(data):
			line = (line-min(line))/(max(line)-min(line))
			# add data to train list
			if foldername == 'MERS':
				x_tensor_list_M.append(line)
				MSample = MSample + 1
				y_tensor_list_M.append([1,0,0,0,0])
			elif foldername == 'SARS':
				x_tensor_list_S.append(line)
				SSample = SSample + 1
				y_tensor_list_S.append([0,1,0,0,0])
			elif foldername == 'SARS-Cov-2':
				x_tensor_list_SC.append(line)
				SCSample = SCSample + 1
				y_tensor_list_SC.append([0,0,1,0,0])
			elif foldername == 'HKU1':
				x_tensor_list_H.append(line)
				HSample = HSample + 1
				y_tensor_list_H.append([0,0,0,1,0])
			elif foldername == 'OC43':
				x_tensor_list_O.append(line)
				OSample = OSample + 1
				y_tensor_list_O.append([0,0,0,0,1])

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

test_acc_MLP = np.zeros((5,train_times))
test_matrix = np.zeros((5,5,train_times))
i = 0
while i < train_times:
	print('Train times:',str(i+1),'/',str(train_times))

	#MLP parameters ###############################################################
	# training parameters
	learning_rate = 0.005
	batch_size = 20
	epoch = 200
	STEPS_PER_EPOCH = len(x_train_tensor_list)/batch_size

	n_hidden_1 = 512
	n_hidden_2 = 256
	n_hidden_3 = 32

	num_classes = 5 # class number
	num_features = np.shape(x_train_tensor_list)[1] # data size

	# load pre-trained model
	model = keras.models.load_model(InputModelName)
	model.summary()
	# Create the base model, removing the last three layers of the previous model
	base_model = tf.keras.models.Sequential(model.layers[:-3])
	base_model.trainable= False
	base_model.build(input_shape=(None,895,))
	base_model.summary()
	# Create the new model
	new_model = tf.keras.Sequential()
	new_model.add(base_model)
	new_model.add(layers.Dense(n_hidden_2, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0)))
	new_model.add(layers.Dense(n_hidden_3, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0)))
	new_model.add(layers.Dense(num_classes, activation='softmax'))
	new_model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
	             loss='categorical_crossentropy',
	             metrics=['categorical_accuracy'])
	new_model.build(input_shape=(None,895,))
	new_model.summary()

	#train
	history1 = new_model.fit(x_train_tensor_list, y_train_tensor_list, batch_size=batch_size, epochs=epoch, verbose=0, validation_split=0.25)
	fig, ax1 = plt.subplots()
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss', color='red')
	ax1.semilogy(history1.history["loss"],label='loss',color='red')
	ax1.semilogy(history1.history["val_loss"],label='val_loss',color='orange')
	ax1.tick_params(axis='y', labelcolor='red')
	#ax1.legend(loc='center right')
	ax2 = ax1.twinx()
	ax2.set_ylabel('Accuracy', color='blue')
	ax2.plot(history1.history["categorical_accuracy"],label='accuracy',color='blue')
	ax2.plot(history1.history["val_categorical_accuracy"],label='val_accuracy',color='green')
	ax2.tick_params(axis='y', labelcolor='blue')
	#ax2.legend(loc='center right')
	#plt.ylim(0.495,0.51)
	fig.tight_layout()
	plt.show()

	# new_model.save(OutputModelName1, save_format='tf')

	# Test dataset classification accuracy ############################################################################################################
	def Predict(x_test_tensor_list):
		y_predict_tensor_list = new_model.predict(x_test_tensor_list)
		j = 0
		P1 = 0
		P2 = 0
		P3 = 0
		P4 = 0
		P5 = 0
		while j < len(y_predict_tensor_list):
			if  y_predict_tensor_list[j,0] == max(y_predict_tensor_list[j]):
				P1 += 1
			elif y_predict_tensor_list[j,1] == max(y_predict_tensor_list[j]):
				P2 += 1
			elif y_predict_tensor_list[j,2] == max(y_predict_tensor_list[j]):
				P3 += 1
			elif y_predict_tensor_list[j,3] == max(y_predict_tensor_list[j]):
				P4 += 1
			elif y_predict_tensor_list[j,4] == max(y_predict_tensor_list[j]):
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
	test_scores1 = new_model.evaluate(x_test_tensor_list_M, y_test_tensor_list_M, verbose=0)
	test_scores2 = new_model.evaluate(x_test_tensor_list_S, y_test_tensor_list_S, verbose=0)
	test_scores3 = new_model.evaluate(x_test_tensor_list_SC, y_test_tensor_list_SC, verbose=0)
	test_scores4 = new_model.evaluate(x_test_tensor_list_H, y_test_tensor_list_H, verbose=0)
	test_scores5 = new_model.evaluate(x_test_tensor_list_O, y_test_tensor_list_O, verbose=0)
	test_acc_MLP[0,i] = (test_scores1[1])
	test_acc_MLP[1,i] = (test_scores2[1])
	test_acc_MLP[2,i] = (test_scores3[1])
	test_acc_MLP[3,i] = (test_scores4[1])
	test_acc_MLP[4,i] = (test_scores5[1])


	i = i+1

print('MLP test acc:', np.mean(test_acc_MLP,axis=1))
print('MLP test acc std:', np.std(test_acc_MLP,axis=1))
print('MLP test matrix:', np.mean(test_matrix,axis=2))

# Print a saliency map
spectra = x_test_tensor_list_M
test_predict = np.zeros((len(spectra),895,5))
n = 0
for line in spectra:
	test = np.zeros((895,895))
	i = 0
	while i < 895:
		test[i] = line
		test[i,i-4] = 0
		test[i,i-3] = 0
		test[i,i-2] = 0
		test[i,i-1] = 0
		test[i,i] = 0
		if i < 894:
			test[i,i+1] = 0
		else:
			test[i,3] = 0
		if i < 893:
			test[i,i+2] = 0
		else:
			test[i,2] = 0
		if i < 892:
			test[i,i+3] = 0
		else:
			test[i,1] = 0
		if i < 891:
			test[i,i+4] = 0
		else:
			test[i,0] = 0
		i += 1
	test_predict[n] += new_model.predict(test)
	n = n+1
test_predict_mean = np.mean(test_predict,axis=0)
test_predict_std = np.std(test_predict,axis=0)
plt.figure(figsize=(8,4),dpi=600)
plt.plot(wavenumber,test_predict_mean[:,0],color='r',label='MERS-CoV')
plt.fill_between(wavenumber,test_predict_mean[:,0]-test_predict_std[:,0],test_predict_mean[:,0]+test_predict_std[:,0],facecolor='r',alpha=0.2)
plt.plot(wavenumber,test_predict_mean[:,1],color='g',label='SARS-CoV')
plt.fill_between(wavenumber,test_predict_mean[:,1]-test_predict_std[:,1],test_predict_mean[:,1]+test_predict_std[:,1],facecolor='g',alpha=0.2)
plt.plot(wavenumber,test_predict_mean[:,2],color='c',label='SARS-CoV-2')
plt.fill_between(wavenumber,test_predict_mean[:,2]-test_predict_std[:,2],test_predict_mean[:,2]+test_predict_std[:,2],facecolor='c',alpha=0.2)
plt.plot(wavenumber,test_predict_mean[:,3],color='b',label='HCoV-HKU1')
plt.fill_between(wavenumber,test_predict_mean[:,3]-test_predict_std[:,3],test_predict_mean[:,3]+test_predict_std[:,3],facecolor='b',alpha=0.2)
plt.plot(wavenumber,test_predict_mean[:,4],color='m',label='HCoV-OC43')
plt.fill_between(wavenumber,test_predict_mean[:,4]-test_predict_std[:,4],test_predict_mean[:,4]+test_predict_std[:,4],facecolor='m',alpha=0.2)
plt.xlabel('Wavenumber / $cm^{-1}$')
plt.ylabel('NN Output')
plt.legend(loc='center left')
plt.show()