import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------
# Dataset Paths
# --------------------------------------

TRAIN_PATH = "casting_data/dataset/train"
TEST_PATH = "casting_data/dataset/test"

IMAGE_SIZE = (224,224)

BATCH_SIZE = 32

# --------------------------------------
# Data Augmentation
# --------------------------------------

train_generator = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True,

    validation_split=0.2

)

test_generator = ImageDataGenerator(
    rescale=1./255
)

# --------------------------------------
# Load Training Images
# --------------------------------------

train_data = train_generator.flow_from_directory(

    TRAIN_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training"

)

validation_data = train_generator.flow_from_directory(

    TRAIN_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation"

)

test_data = test_generator.flow_from_directory(

    TEST_PATH,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False

)

# --------------------------------------
# CNN Model
# --------------------------------------

model = Sequential([

Conv2D(32,(3,3),activation="relu",input_shape=(224,224,3)),

MaxPooling2D(),

Conv2D(64,(3,3),activation="relu"),

MaxPooling2D(),

Conv2D(128,(3,3),activation="relu"),

MaxPooling2D(),

Flatten(),

Dense(128,activation="relu"),

Dropout(0.5),

Dense(1,activation="sigmoid")

])

# --------------------------------------
# Compile
# --------------------------------------

model.compile(

optimizer="adam",

loss="binary_crossentropy",

metrics=["accuracy"]

)

# --------------------------------------
# Early Stopping
# --------------------------------------

early_stop = EarlyStopping(

monitor="val_loss",

patience=3,

restore_best_weights=True

)

# --------------------------------------
# Train
# --------------------------------------

history = model.fit(

train_data,

validation_data=validation_data,

epochs=15,

callbacks=[early_stop]

)

# --------------------------------------
# Evaluate
# --------------------------------------

loss,accuracy = model.evaluate(test_data)

print("\nTest Accuracy:",accuracy)

# --------------------------------------
# Save Model
# --------------------------------------

model.save("defect_model.keras")

# --------------------------------------
# Save Results
# --------------------------------------

with open("results.txt","w") as file:

    file.write(f"Test Accuracy : {accuracy:.4f}\n")

# --------------------------------------
# Accuracy Plot
# --------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"],label="Training")

plt.plot(history.history["val_accuracy"],label="Validation")

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("training_accuracy.png")

plt.show()

# --------------------------------------
# Loss Plot
# --------------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"],label="Training")

plt.plot(history.history["val_loss"],label="Validation")

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.savefig("training_loss.png")

plt.show()

