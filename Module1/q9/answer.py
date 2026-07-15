import time
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# ------------------------------------------
# Configuration
# ------------------------------------------

IMAGE_SIZE = (224,224)

BATCH_SIZE = 32

DATASET_PATH = "data"

# ------------------------------------------
# Data Generator
# ------------------------------------------

datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True

)

train_data = datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    subset="training",

    class_mode="binary"

)

validation_data = datagen.flow_from_directory(

    DATASET_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    subset="validation",

    class_mode="binary",

    shuffle=False

)

# ------------------------------------------
# Load MobileNet
# ------------------------------------------

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(224,224,3)

)

# Freeze all layers

base_model.trainable = False

# ------------------------------------------
# Add New Classifier
# ------------------------------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

x = Dense(128,activation="relu")(x)

output = Dense(1,activation="sigmoid")(x)

model = Model(base_model.input,output)

# ------------------------------------------
# Compile
# ------------------------------------------

model.compile(

optimizer=Adam(learning_rate=0.001),

loss="binary_crossentropy",

metrics=["accuracy"]

)

early_stop = EarlyStopping(

monitor="val_loss",

patience=3,

restore_best_weights=True

)

# ------------------------------------------
# Stage 1 Training
# ------------------------------------------

history = model.fit(

train_data,

validation_data=validation_data,

epochs=5,

callbacks=[early_stop]

)

# ------------------------------------------
# Fine Tuning
# ------------------------------------------

base_model.trainable = True

for layer in base_model.layers[:-20]:

    layer.trainable=False

model.compile(

optimizer=Adam(learning_rate=1e-5),

loss="binary_crossentropy",

metrics=["accuracy"]

)

history_fine = model.fit(

train_data,

validation_data=validation_data,

epochs=5,

callbacks=[early_stop]

)

# ------------------------------------------
# Prediction
# ------------------------------------------

start=time.time()

pred=model.predict(validation_data)

end=time.time()

prediction_time=(end-start)/len(pred)

pred=(pred>0.5).astype(int)

true=validation_data.classes

precision=precision_score(true,pred)

recall=recall_score(true,pred)

f1=f1_score(true,pred)

print("Precision :",precision)

print("Recall :",recall)

print("F1 :",f1)

print("Inference Time :",prediction_time,"seconds/image")

print()

print(classification_report(true,pred))

# ------------------------------------------
# Save Results
# ------------------------------------------

with open("results.txt","w") as file:

    file.write(f"Precision : {precision:.4f}\n")

    file.write(f"Recall : {recall:.4f}\n")

    file.write(f"F1 Score : {f1:.4f}\n")

    file.write(f"Inference Time : {prediction_time:.6f} sec/image\n")

# ------------------------------------------
# Save Model
# ------------------------------------------

model.save("mask_detector.keras")

# ------------------------------------------
# Accuracy Plot
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"]+history_fine.history["accuracy"],label="Training")

plt.plot(history.history["val_accuracy"]+history_fine.history["val_accuracy"],label="Validation")

plt.legend()

plt.title("Accuracy")

plt.savefig("training_accuracy.png")

plt.show()

# ------------------------------------------
# Loss Plot
# ------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"]+history_fine.history["loss"],label="Training")

plt.plot(history.history["val_loss"]+history_fine.history["val_loss"],label="Validation")

plt.legend()

plt.title("Loss")

plt.savefig("training_loss.png")

plt.show()

