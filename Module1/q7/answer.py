import os
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split

# -------------------------------------------------
# STEP 1
# Dataset Path
# -------------------------------------------------

DATASET_PATH = "dataset"

# Image size

IMAGE_SIZE = (224,224)

# Empty list

data=[]

# -------------------------------------------------
# STEP 2
# Read each folder
# -------------------------------------------------

classes=os.listdir(DATASET_PATH)

print("Classes Found")

print(classes)

# -------------------------------------------------
# STEP 3
# Read every image
# -------------------------------------------------

for class_name in classes:

    class_folder=os.path.join(DATASET_PATH,class_name)

    if not os.path.isdir(class_folder):
        continue

    for image_name in os.listdir(class_folder):

        image_path=os.path.join(class_folder,image_name)

        image=cv2.imread(image_path)

        if image is None:
            continue

        # Resize

        image=cv2.resize(image,IMAGE_SIZE)

        # Normalize

        image=image/255.0

        # Instead of storing huge image

        # Store only metadata

        data.append({

            "Image Path":image_path,

            "Class":class_name,

            "Width":IMAGE_SIZE[0],

            "Height":IMAGE_SIZE[1]

        })

# -------------------------------------------------
# STEP 4
# Create DataFrame
# -------------------------------------------------

metadata=pd.DataFrame(data)

print()

print(metadata.head())

print()

print("Total Images :",len(metadata))

# -------------------------------------------------
# STEP 5
# Split Dataset
# -------------------------------------------------

train_df,val_df=train_test_split(

metadata,

test_size=0.2,

random_state=42,

stratify=metadata["Class"]

)

# -------------------------------------------------
# STEP 6
# Save CSV
# -------------------------------------------------

metadata.to_csv("metadata.csv",index=False)

train_df.to_csv("train.csv",index=False)

val_df.to_csv("validation.csv",index=False)

print()

print("Metadata Saved")

print("Training Images :",len(train_df))

print("Validation Images :",len(val_df))

print()

print("Pipeline Completed Successfully")