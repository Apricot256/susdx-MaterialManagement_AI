from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from  tensorflow_hub import KerasLayer

def getVGG16(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    return model

def getEfficientNet(input_shape, num_classes):

    model = Sequential(
        [
            KerasLayer(
                "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", # B0
                # "https://tfhub.dev/tensorflow/efficientnet/b1/feature-vector/1", # B1
                # "https://tfhub.dev/tensorflow/efficientnet/b2/feature-vector/1", # B2
                # "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1", # B3
                # "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1", # B4
                # "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1", # B5
                # "https://tfhub.dev/tensorflow/efficientnet/b6/feature-vector/1", # B6
                # "https://tfhub.dev/tensorflow/efficientnet/b7/feature-vector/1", # B7
                trainable=False,
            ),
            Dense(num_classes, activation="sigmoid"),
        ]
    )
    model.build([None, input_shape[0], input_shape[1], 3])
    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model