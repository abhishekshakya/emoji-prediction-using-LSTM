import emoji
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint

emoji_dict = {
	0:"♥",
	1:"⚾",
	2:"😃",
	3:"😥",
	4:"🍴",
	5:"💯",
	6:"🔥",
	7:"😘",
	8:"🌰",
	9:"💪",
}
import pandas as pd 
import numpy as np 

df_train = pd.read_csv('train_emoji.csv')
df_test = pd.read_csv('test_emoji.csv')

# print(df_train.head())

X_train = df_train.iloc[:,0].values
Y_train = df_train.iloc[:,1].values

Y_train = to_categorical(Y_train,num_classes=5)

X_test = df_test.iloc[:,0].values
Y_test = df_test.iloc[:,1].values

Y_test = to_categorical(Y_test,num_classes=5)

# print(X_train,Y_train)

# for i in range(5):
# 	print(X_train[i],emoji_dict[Y_train[i]])

f = open('glove.6B.50d.txt',encoding='utf8')#50d = 50 dimesional


embeddings_index = {}
for line in f:
	values = line.split()

	word = values[0]
	coeffs = np.asarray(values[1:],dtype='float')

	embeddings_index[word] =  coeffs

f.close()

emd_dim = embeddings_index['eat'].shape[0]#dimension of embedding (this case we have 50)


def embedding_output(X):
	max_len = 10#number of RNN cells or number of words in sentence

	embedding_out =  np.zeros((X.shape[0],max_len,emd_dim))

	for i in range(X.shape[0]):
		X[i] = X[i].split()
		for j in range(len(X[i])):
				embedding_out[i][j] = embeddings_index[X[i][j].lower()]

	return embedding_out


embedding_matrix_train = embedding_output(X_train)
embedding_matrix_test = embedding_output(X_test)




model = Sequential()
model.add(LSTM(64,input_shape=(10,50),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64,return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# checkpoint = ModelCheckpoint("best_model.h5",monitor='val_loss',save_best_only=True)
earlystop = EarlyStopping(monitor='val_acc',patience=10)

hist = model.fit(embedding_matrix_train,Y_train,epochs=100,batch_size=64,shuffle=True,validation_split=0.2)

###############stacked LSTM(complex LSTM)/(multiple layers of LSTM)

# model.load_weights("best_model.h5")

print(model.evaluate(embedding_matrix_test,Y_test))

pred = model.predict_classes(embedding_matrix_test)

for i in range(len(pred)):
	print(" ".join(X_test[i]))
	# print(df_test.iloc[1,1])
	print(emoji_dict[pred[i]],emoji_dict[df_test.iloc[i,1]])
	print("****************************************************")








