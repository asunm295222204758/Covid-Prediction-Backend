import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/"
CATEGORIES = ["COVID","Lung_Opacity", "Normal", "Viral Pneumonia"]
IMG_SIZE = 224  

def load_dataset():
    images, labels = [], []
    
    for label, category in enumerate(CATEGORIES):
        folder_path = os.path.join(DATASET_PATH, category)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  
            images.append(img)
            labels.append(label)

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)
    
    return images, labels

X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save("dataset/X_train.npy", X_train)
np.save("dataset/X_test.npy", X_test)
np.save("dataset/y_train.npy", y_train)
np.save("dataset/y_test.npy", y_test)

print("Dataset prepared successfully!")
