import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

# from keras.engine.sequential import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

import pickle

# ############################################################

# Part 1 :---

# Importing the dataset....

path = 'myData'
test_ratio = 0.2
validation_ratio = 0.2
imageDimension = (32, 32, 3)

batchSizeVal = 50
epochsVal = 1
stepsPerEpoch = 6502 // batchSizeVal

images = []
classNo = []
myList = os.listdir(path)
print("Total No. of Classes Detected", len(myList))

no_of_classes = len(myList)
print("Importing Classes.....")

for x in range(0, no_of_classes):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (imageDimension[0], imageDimension[1]))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
print("\nLength of images: " + str(len(images)))

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)

# ############################################################

# Part 2 :---

# Splitting the dataset....

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=test_ratio, random_state=0)

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                test_size=validation_ratio,
                                                                random_state=0)

print("Training Data Shape: ", X_train.shape)
print("test Data Shape: ", X_test.shape)
print("Validation Data Shape: ", X_validation.shape)

no_of_samples = []
for x in range(0, no_of_classes):
    # print(np.where(y_train == 0)[0])
    no_of_samples.append(len(np.where(y_train == 0)[0]))
print(no_of_samples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, no_of_classes), no_of_samples)
plt.title("No. of images for each class")
plt.xlabel("Class ID")
plt.ylabel("No. of Images")


# plt.show()


def preProcessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255
    return image


# img = preProcessing(X_train[30])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("Preprocessed Image", img)
# cv2.waitKey(0)


# X_train = np.array(list(map(preProcessing, X_train)))
# print(X_train[30].shape)
# img = X_train[30]
# img = cv2.resize(img, (300, 300))
# cv2.imshow("Preprocessed Image", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

print("Shape of X_train before: ", X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
print("Shape of X_train after: ", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)

# OneHotEncoding

y_train = to_categorical(y_train, no_of_classes)
y_validation = to_categorical(y_validation, no_of_classes)
y_test = to_categorical(y_test, no_of_classes)


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()
    model.add(
        (Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimension[0], imageDimension[1], 1), activation='relu')))

    model.add(
        (Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(
        (Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(
        (Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_of_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())

history = model.fit(dataGen.flow(X_train, y_train, batch_size=batchSizeVal), steps_per_epoch=stepsPerEpoch,
                    epochs=epochsVal, validation_data=(X_validation, y_validation),
                    shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])

pickle_out = open("A/model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
