import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.backend import ctc_batch_cost,reverse
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import numpy as np
from Global_parameter import *
from Data_Generation import Generation
from os.path import exists
from os import listdir
import cv2

class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return y_pred

class Reverse_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return reverse(inputs,axes=1)

class Net_Model(Model):
    def __init__(self,input_shape=None,training=True):
        # input
        super().__init__()
        self.inputs_shape = input_shape
        self.train_flag = training
        self.generation = Generation()
        self.My_Loss_Function = MeanSquaredError()
        self.__build__()

    def __build__(self):

        # Convolution layer (VGG)
        self.C1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')  # (None, 128, 64, 64)
        self.B1 = BatchNormalization()
        self.A1 = Activation('relu')
        self.P1 = MaxPooling2D(pool_size=(2, 2))  # (None,64, 32, 64)

        self.C2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')  # (None, 64, 32, 128)
        self.B2 = BatchNormalization()
        self.A2 = Activation('relu')
        self.P2 = MaxPooling2D(pool_size=(2, 2))  # (None, 32, 16, 128)

        self.C3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')  # (None, 32, 16, 256)
        self.B3 = BatchNormalization()
        self.A3 = Activation('relu')

        self.C4 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')  # (None, 32, 16, 256)
        self.B4 = BatchNormalization()
        self.A4 = Activation('relu')
        self.P4 = MaxPooling2D(pool_size=(1, 2))  # (None, 32, 8, 256)

        self.C5 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')  # (None, 32, 8, 512)
        self.B5 = BatchNormalization()
        self.A5 = Activation('relu')

        self.C6 = Conv2D(512, (3, 3), padding='same')  # (None, 32, 8, 512)
        self.B6 = BatchNormalization()
        self.A6 = Activation('relu')
        self.P6 = MaxPooling2D(pool_size=(1, 2))  # (None, 32, 4, 512)

        self.C7 = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal')  # (None, 32, 4, 512)
        self.B7 = BatchNormalization()
        self.A7 = Activation('relu')

        # CNN to RNN
        self.R8 = Reshape(target_shape=((32, 10240)))  # (None, 32, 2048)
        self.F8 = Dense(64, activation='relu', kernel_initializer='he_normal')  # (None, 32, 64)

        # RNN layer
        self.L9_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal')  # (None, 32, 512)
        self.L9_2 = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')
        self.V9   = Reverse_Layer()
        self.D9   = add                        # (None, 32, 512)
        self.B9   = BatchNormalization()

        self.L10_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal')
        self.L10_2 = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')
        self.V10   = Reverse_Layer()
        self.D10   = concatenate               # (None, 32, 1024)
        self.B10   = BatchNormalization()

        # transforms RNN output to character activations:
        self.F11 = Dense(num_classes, kernel_initializer='he_normal')  # (None, 32, 63)
        self.A11 = Activation('softmax')

        # loss
        self.loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,))

    # 全部参考于github上韩文车牌识别项目的CRNN网络结构
    # # Loss and train functions, network architecture
    def __Input__(self,name:str, tensor=None, dtype=None, shape=None):
        if tensor:
            if name:
                input = tensor[name]
            else:
                input = tensor
            if dtype:
                input = tf.cast(input,dtype)
        return input

    def ctc_lambda_func(self,args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]

        return ctc_batch_cost(labels, y_pred, input_length, label_length)

    def call(self,input):

        if self.train_flag:
            inputs = self.__Input__(tensor=input, name='the_input', dtype='float32')  # (None, 128, 64, 1)
            labels = self.__Input__(tensor=input, name='the_labels', dtype='float32')  # (None ,8)
            input_length = self.__Input__(tensor=input, name='input_length', dtype='int64')  # (None, 1)
            label_length = self.__Input__(tensor=input, name='label_length', dtype='int64')  # (None, 1)
        else:
            inputs = input

        input = self.C1(inputs)
        input = self.B1(input)
        input = self.A1(input)
        input = self.P1(input)

        input = self.C2(input)
        input = self.B2(input)
        input = self.A2(input)
        input = self.P2(input)

        input = self.C3(input)
        input = self.B3(input)
        input = self.A3(input)

        input = self.C4(input)
        input = self.B4(input)
        input = self.A4(input)
        input = self.P4(input)

        input = self.C5(input)
        input = self.B5(input)
        input = self.A5(input)

        input = self.C6(input)
        input = self.B6(input)
        input = self.A6(input)
        input = self.P6(input)

        input = self.C7(input)
        input = self.B7(input)
        input = self.A7(input)
        # Convolution layer end (VGG)

        # CNN to RNN
        input = self.R8(input)
        input = self.F8(input)

        # RNN layer
        lstm_1  = self.L9_1(input)
        lstm_1b = self.L9_2(input)
        input = self.V9(lstm_1b)
        input = self.D9([lstm_1,input])
        input = self.B9(input)

        lstm_2  = self.L10_1(input)
        lstm_2b = self.L10_2(input)
        input = self.V10(lstm_2b)
        input = self.D10([lstm_2, input])
        input = self.B10(input)

        # transforms RNN output to character activations:
        input = self.F11(input)
        input = self.A11(input)

        if self.train_flag:
            self.loss_out([input, labels, input_length, label_length])
            return self.loss_out([input, labels, input_length, label_length])
        else:
            return input

    def __load_weight__(self):
        self.checkpoint_save_path = "./checkpoint/CRNN.ckpt"
        if exists(self.checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            self.load_weights(self.checkpoint_save_path)

    def train(self):

        self.compile(
            loss=self.My_Loss_Function,
            optimizer='adam',
        )

        self.__load_weight__()

        cp_callback = ModelCheckpoint(filepath=self.checkpoint_save_path,
                                                         monitor='loss',
                                                         verbose=1,
                                                         mode='min',
                                                         period=1)

        self.history = self.fit(
            x = self.__load_data__(),
            epochs = epochs,
            steps_per_epoch=int(n / batch_size),
            callbacks=[cp_callback]
        )

        self.summary()

        # print(model.trainable_variables)
        with open('../weights.txt', 'w', encoding='utf-8') as file:
            for v in self.trainable_variables:
                file.write(str(v.name) + '\n')
                file.write(str(v.shape) + '\n')
                file.write(str(v.numpy()) + '\n')

    def __load_data__(self):
        return self.generation.next_batch()

    def easy_predict(self,img):
        self.__load_weight__()
        img = cv2.resize(img, (img_h, img_w))
        img = np.rot90(img)
        img = img.astype(np.float32)
        img = np.transpose(img, [1, 0, 2])
        img_pred = (img / 255.0) * 2.0 - 1.0
        img_pred = np.expand_dims(img_pred, axis=0)
        net_out_value = self.predict(img_pred)
        pred_texts = self.generation.decode_label(net_out_value)
        print('Predicted: %s' % (pred_texts))
        return pred_texts

    def draw(self):
        # 显示训练集和验证集的acc和loss曲线
        acc = self.history.history['sparse_categorical_accuracy']
        val_acc = self.history.history['val_sparse_categorical_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    input_shape = (img_w, img_h, img_c)  # (128, 64, 1)
    np.set_printoptions(threshold=np.inf)
    model = Net_Model(
        input_shape = input_shape,
        training = False
    )
    
    # model.train()
    img_files = listdir(Train_file)
    for img_path in img_files:
        img = cv2.imread(Train_file + img_path)
        print(Train_file + img_path)
        model.easy_predict(img)
