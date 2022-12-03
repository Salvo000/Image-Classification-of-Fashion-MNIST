import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
# from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def __load_fascion_mnist():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    print('Fashion MNIST Dataset Shape:')
    print('X_train: ' + str(X_train.shape))
    print('Y_train: ' + str(Y_train.shape))
    print('X_test:  '  + str(X_test.shape))
    print('Y_test:  '  + str(Y_test.shape))
    return (X_train, Y_train), (X_test, Y_test)

def __read_samples( x,y , num_row=4,num_col=8):
    # get a segment of the dataset
    num = num_row*num_col
    images = x[:num]
    labels = y[:num]

    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()

def __create_net():

    inputs = keras.Input( shape=(28,28,1))
    x = layers.Conv2D( 32, (3,3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense( 64, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    return keras.Model(inputs,outputs,name="CNN" )

def __to_categorical( labels: np.array ) -> np.array :
    classes_number = max( labels )+ 1
    length = len( labels )
    y = np.zeros(( length,classes_number  ))
    for pos,l in enumerate(labels):
        y[pos, l] = 1

    return y



if __name__ == '__main__':
    # remove this line to activate gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    load_weights = True
    do_train = False
    do_test = True


    (x_train,y_train), (x_test,y_test) = __load_fascion_mnist()

    x_train = tf.convert_to_tensor( x_train )
    encoded_y_train = tf.convert_to_tensor(__to_categorical( y_train))

    print("Data after preprocessing")
    print('X_train: ' + str(x_train.shape))
    print('encoded_y_train: ' + str(encoded_y_train.shape))

    model = __create_net()

    if load_weights:
        model.load_weights("./wgt/model.h5")

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr = 0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    if do_train:

        history = model.fit(x_train, encoded_y_train, epochs=10, batch_size=64, validation_split=0.2 )
        model.save_weights("./wgt/model.h5")

        plt.plot(history.history['loss'], label="train_loss")
        plt.plot(history.history['val_loss'],label="val_loss")
        plt.plot(history.history['accuracy'],label="accuracy")
        plt.plot(history.history['val_accuracy'],label="val_accuracy")

        plt.title("Train vs Accuracy History")
        plt.ylabel('loss/accuracy')
        plt.xlabel('epoch')
        plt.legend()

        plt.show()



    if do_test:
        x_test = tf.convert_to_tensor(x_test)
        encoded_y_test = tf.convert_to_tensor(__to_categorical( y_test ))
        model.evaluate( x_test, encoded_y_test )

        res = model.predict( x_test )
        preds = np.argmax( res, axis=1 )
        print( preds)
        print( y_test)

        # make confusion matrix
        cm = confusion_matrix(y_test,preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()


    # __read_samples(x_train,y_train)
    # #how many labels ?
    # num,freq = np.unique( y_test, return_counts=True)
    # print(dict(zip(num,freq)))
