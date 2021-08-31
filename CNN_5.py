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

OutputModelName2 = 'D:\\Model\\CNN_5'

data_length = 895
train_data_ratio = 0.8
train_times = 1

# 训练集和测试集处理##########################################################################################################
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

test_acc_CNN = np.zeros((5,train_times))
test_matrix = np.zeros((5,5,train_times))
i = 0
while i < train_times:
	print('Train times:',str(i+1),'/',str(train_times))

	# CNN parameters ################################################################
	# training parameters
	#learning_rate = 0.01
	batch_size = 50
	epoch = 100
	STEPS_PER_EPOCH = len(x_train_tensor_list)/batch_size

	lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
	  0.01,
	  decay_steps=STEPS_PER_EPOCH*50,
	  decay_rate=1,
	  staircase=False)
	step = np.linspace(0,STEPS_PER_EPOCH*epoch)
	lr = lr_schedule(step)
	plt.figure(figsize = (8,6))
	plt.plot(step/STEPS_PER_EPOCH, lr)
	plt.ylim([0,max(plt.ylim())])
	plt.xlabel('Epoch')
	plt.ylabel('Learning Rate')
	learning_rate = lr_schedule

	num_classes = 5 # class number
	num_features = np.shape(x_train_tensor_list)[1] # data size

	# 1D LeNet-5
	model2 = tf.keras.Sequential()
	model2.add(layers.InputLayer(input_shape=(num_features,1), name='input'))
	model2.add(layers.Conv1D(filters=32, kernel_size=10, padding='same'))
	model2.add(layers.BatchNormalization())
	model2.add(layers.Activation('relu'))
	model2.add(layers.AvgPool1D(pool_size=2, strides=2))
	model2.add(layers.Conv1D(filters=64, kernel_size=5))
	model2.add(layers.BatchNormalization())
	model2.add(layers.Activation('relu'))
	model2.add(layers.AvgPool1D(pool_size=2, strides=2))
	model2.add(layers.Flatten())
	model2.add(layers.Dense(128, activation='relu'))
	model2.add(layers.Dropout(0.2))
	model2.add(layers.Dense(32, activation='relu'))
	model2.add(layers.Dropout(0.2))
	model2.add(layers.Dense(num_classes, activation='softmax'))
	model2.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
	             loss='categorical_crossentropy',
	             metrics=['categorical_accuracy'])
	model2.summary()
	# keras.utils.plot_model(model, "mpl_model.png", show_shapes=True)

	x_train_tensor_list = np.reshape(x_train_tensor_list,(np.shape(x_train_tensor_list)[0],np.shape(x_train_tensor_list)[1],1))
	y_train_tensor_list = np.reshape(y_train_tensor_list,(np.shape(y_train_tensor_list)[0],np.shape(y_train_tensor_list)[1],1))
	#train
	history2 = model2.fit(x_train_tensor_list, y_train_tensor_list, batch_size=batch_size, epochs=epoch, verbose=0, validation_split=0.25)
	fig, ax1 = plt.subplots()
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss', color='red')
	ax1.semilogy(history2.history["loss"],label='loss',color='red')
	ax1.semilogy(history2.history["val_loss"],label='val_loss',color='orange')
	ax1.tick_params(axis='y', labelcolor='red')
	#ax1.legend(loc='center right')
	ax2 = ax1.twinx()
	ax2.set_ylabel('Accuracy', color='blue')
	ax2.plot(history2.history["categorical_accuracy"],label='accuracy',color='blue')
	ax2.plot(history2.history["val_categorical_accuracy"],label='val_accuracy',color='green')
	ax2.tick_params(axis='y', labelcolor='blue')
	#ax2.legend(loc='center right')
	#plt.ylim(0.495,0.51)
	fig.tight_layout()
	plt.show()

	# model2.save(OutputModelName1, save_format='tf')

	# Test dataset classification accuracy ##############################################################################################################
	x_test_tensor_list_M = np.reshape(x_test_tensor_list_M,(np.shape(x_test_tensor_list_M)[0],np.shape(x_test_tensor_list_M)[1],1))
	x_test_tensor_list_S = np.reshape(x_test_tensor_list_S,(np.shape(x_test_tensor_list_S)[0],np.shape(x_test_tensor_list_S)[1],1))
	x_test_tensor_list_SC = np.reshape(x_test_tensor_list_SC,(np.shape(x_test_tensor_list_SC)[0],np.shape(x_test_tensor_list_SC)[1],1))
	x_test_tensor_list_H = np.reshape(x_test_tensor_list_H,(np.shape(x_test_tensor_list_H)[0],np.shape(x_test_tensor_list_H)[1],1))
	x_test_tensor_list_O = np.reshape(x_test_tensor_list_O,(np.shape(x_test_tensor_list_O)[0],np.shape(x_test_tensor_list_O)[1],1))

	def Predict(x_test_tensor_list):
		y_predict_tensor_list = model2.predict(x_test_tensor_list)
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
	test_scores1 = model2.evaluate(x_test_tensor_list_M, y_test_tensor_list_M, verbose=0)
	test_scores2 = model2.evaluate(x_test_tensor_list_S, y_test_tensor_list_S, verbose=0)
	test_scores3 = model2.evaluate(x_test_tensor_list_SC, y_test_tensor_list_SC, verbose=0)
	test_scores4 = model2.evaluate(x_test_tensor_list_H, y_test_tensor_list_H, verbose=0)
	test_scores5 = model2.evaluate(x_test_tensor_list_O, y_test_tensor_list_O, verbose=0)
	test_acc_CNN[0,i] = (test_scores1[1])
	test_acc_CNN[1,i] = (test_scores2[1])
	test_acc_CNN[2,i] = (test_scores3[1])
	test_acc_CNN[3,i] = (test_scores4[1])
	test_acc_CNN[4,i] = (test_scores5[1])


	i = i+1

print('CNN test acc:', np.mean(test_acc_CNN,axis=1))
print('CNN test acc std:', np.std(test_acc_CNN,axis=1))
print('CNN test matrix:', np.mean(test_matrix,axis=2))
