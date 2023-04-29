import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score , confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the saved face data
data = np.load("./face_data.npy")
X = data[:, 1:].astype(np.uint8)
y = data[:, 0]

# Reshape X to 4D array (number of images, height, width, channels)
X = X.reshape((-1, 100, 100, 1))

# Normalize the pixel values
X = X / 255.0

# Convert y to one-hot encoding
# One-hot encode labels
le = LabelEncoder()
y = le.fit_transform(y)
num_classes = len(le.classes_)
y = to_categorical(y, num_classes=num_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes, average='macro')
cm = confusion_matrix(y_test_classes,y_pred_classes)

print("Accuracy:", accuracy)
print("F1 score:", f1)
print("cm :",cm)
