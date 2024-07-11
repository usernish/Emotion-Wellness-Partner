# model.py

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utils import load_fer2013_dataset, preprocess_images

def train_model(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_flat, y_train)
    return model
