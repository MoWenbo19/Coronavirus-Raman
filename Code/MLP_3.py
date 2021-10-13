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

# data set path defination
data_path_Train = 'D:\\Data\\3-class'

OutputModelName1 = 'D:\\Model\\MLP_3'

data_length = 895
train_data_ratio = 1605
train_times = 10

# Training and test datasets ##########################################################################################################
# load data
x_tensor_list_M = [] #x-features, y-label
y_tensor_list_M = []
x_tensor_list_S = []
y_tensor_list_S = []
x_tensor_list_SC = []
y_tensor_list_SC = []

file_count = 0
MSample = 0
SSample = 0
SCSample = 0
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
				y_tensor_list_M.append([1,0,0])
			elif foldername == 'SARS':
				x_tensor_list_S.append(line)
				SSample = SSample + 1
				y_tensor_list_S.append([0,1,0])
			elif foldername == 'SARS-Cov-2':
				x_tensor_list_SC.append(line)
				SCSample = SCSample + 1
				y_tensor_list_SC.append([0,0,1])


x_tensor_list_M = np.asarray(x_tensor_list_M)
y_tensor_list_M = np.asarray(y_tensor_list_M)
x_tensor_list_S = np.asarray(x_tensor_list_S)
y_tensor_list_S = np.asarray(y_tensor_list_S)
x_tensor_list_SC = np.asarray(x_tensor_list_SC)
y_tensor_list_SC = np.asarray(y_tensor_list_SC)


print('MERS sample in data : ',str(MSample))
print('SARS sample in data : ',str(SSample))
print('SARS-Cov-2 sample in data : ',str(SCSample))

per = np.random.permutation(x_tensor_list_M.shape[0])
x_tensor_list_M = x_tensor_list_M[per,:]
x_train_tensor_list_M = np.asarray(x_tensor_list_M[0:train_data_ratio])
x_test_tensor_list_M = np.asarray(x_tensor_list_M[train_data_ratio:])
y_train_tensor_list_M = np.asarray(y_tensor_list_M[0:train_data_ratio])
y_test_tensor_list_M = np.asarray(y_tensor_list_M[train_data_ratio:])

per = np.random.permutation(x_tensor_list_S.shape[0])
x_tensor_list_S = x_tensor_list_S[per,:]
x_train_tensor_list_S = np.asarray(x_tensor_list_S[0:train_data_ratio])
x_test_tensor_list_S = np.asarray(x_tensor_list_S[train_data_ratio:])
y_train_tensor_list_S = np.asarray(y_tensor_list_S[0:train_data_ratio])
y_test_tensor_list_S = np.asarray(y_tensor_list_S[train_data_ratio:])

per = np.random.permutation(x_tensor_list_SC.shape[0])
x_tensor_list_SC = x_tensor_list_SC[per,:]
x_train_tensor_list_SC = np.asarray(x_tensor_list_SC[0:train_data_ratio])
x_test_tensor_list_SC = np.asarray(x_tensor_list_SC[train_data_ratio:])
y_train_tensor_list_SC = np.asarray(y_tensor_list_SC[0:train_data_ratio])
y_test_tensor_list_SC = np.asarray(y_tensor_list_SC[train_data_ratio:])

x_train_tensor_list = np.vstack((x_train_tensor_list_M,x_train_tensor_list_S,x_train_tensor_list_SC))
y_train_tensor_list = np.vstack((y_train_tensor_list_M,y_train_tensor_list_S,y_train_tensor_list_SC))
per = np.random.permutation(x_train_tensor_list.shape[0])
x_train_tensor_list = x_train_tensor_list[per,:]
y_train_tensor_list = y_train_tensor_list[per,:]


test_acc_MLP = np.zeros((3,train_times))
test_matrix = np.zeros((3,3,train_times))
i = 0
while i < train_times:
	print('Train times:',str(i+1),'/',str(train_times))

	# MLP parameters################################################################
	# training parameters
	learning_rate = 0.01
	batch_size = 50
	epoch = 200

	n_hidden_1 = 512
	n_hidden_2 = 256
	n_hidden_3 = 32

	num_classes = 3 # class number
	num_features = np.shape(x_train_tensor_list)[1] # data size

	#build mpl model
	inputs = tf.keras.Input(shape=num_features, name='input') #输入层，形状为num_features*1
	h1 = layers.Dense(n_hidden_1, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0))(inputs)
	h2 = layers.Dense(n_hidden_2, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0))(h1)
	h3 = layers.Dense(n_hidden_3, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0))(h2)
	outputs = layers.Dense(num_classes, activation='softmax')(h3) #输出层
	model1 = tf.keras.Model(inputs=inputs, outputs=outputs, name='mpl_model')
	model1.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
	             loss='categorical_crossentropy',
	             metrics=['categorical_accuracy'])
	model1.summary()
	# keras.utils.plot_model(model, "mpl_model.png", show_shapes=True)

	#train
	history1 = model1.fit(x_train_tensor_list, y_train_tensor_list, batch_size=batch_size, epochs=epoch, verbose=0, validation_split=0.25)
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

	# model1.save(OutputModelName1, save_format='tf')

	# Test dataset classification accuracy #########################################################################################################
	def Predict(x_test_tensor_list):
		y_predict_tensor_list = model1.predict(x_test_tensor_list)
		j = 0
		P1 = 0
		P2 = 0
		P3 = 0
		while j < len(y_predict_tensor_list):
			if  y_predict_tensor_list[j,0] == max(y_predict_tensor_list[j]):
				P1 += 1
			elif y_predict_tensor_list[j,1] == max(y_predict_tensor_list[j]):
				P2 += 1
			elif y_predict_tensor_list[j,2] == max(y_predict_tensor_list[j]):
				P3 += 1
			j = j+1
		return np.array([P1,P2,P3])/len(y_predict_tensor_list)*100

	print(Predict(x_test_tensor_list_M))
	print(Predict(x_test_tensor_list_S))
	print(Predict(x_test_tensor_list_SC))
	test_matrix[0,:,i] = Predict(x_test_tensor_list_M)
	test_matrix[1,:,i] = Predict(x_test_tensor_list_S)
	test_matrix[2,:,i] = Predict(x_test_tensor_list_SC)
	test_scores1 = model1.evaluate(x_test_tensor_list_M, y_test_tensor_list_M, verbose=0)
	test_scores2 = model1.evaluate(x_test_tensor_list_S, y_test_tensor_list_S, verbose=0)
	test_scores3 = model1.evaluate(x_test_tensor_list_SC, y_test_tensor_list_SC, verbose=0)
	test_acc_MLP[0,i] = (test_scores1[1])
	test_acc_MLP[1,i] = (test_scores2[1])
	test_acc_MLP[2,i] = (test_scores3[1])


	i = i+1

print('MLP test acc:', np.mean(test_acc_MLP,axis=1))
print('MLP test acc std:', np.std(test_acc_MLP,axis=1))
print('MLP test matrix:', np.mean(test_matrix,axis=2))
