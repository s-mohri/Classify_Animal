# USAGE
# python shallownet_animals.py --dataset ../datasets/animals
# py -m shallownet_animals --dataset ../datasets/animals

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.preprocessing import ImageToArrayPreprocessor
from utils.preprocessing import SimpleImageResizer
from utils.dataset import SimpleDatasetLoader
from utils.nn.conv import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, help="path to input dataset")
args = vars(ap.parse_args())
print("[INFO] loading image...")
imagePaths = list(paths.list_images(args["dataset"]))
sp = SimpleImageResizer(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))
data = data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_cross entropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="training_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="validation_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="training_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="validation_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
