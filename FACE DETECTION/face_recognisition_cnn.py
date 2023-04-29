import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Initialize the video capture and the face detector
cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if ret:
        # Detect faces in the frame
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face

            cut = frame[y:y + h, x:x + w]

            fix = cv2.resize(cut, (100, 100))

            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)
            X_test = gray.astype(np.uint8).reshape(-1, 100, 100, 1) / 255.0  # Normalize pixel values
            y_pred = model.predict(X_test)
            label = le.inverse_transform(np.argmax(y_pred, axis=1))[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)


        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
