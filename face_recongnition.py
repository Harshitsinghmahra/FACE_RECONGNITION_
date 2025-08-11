import cv2
import numpy as np
import os
import shutil
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from collections import deque

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Enhanced face tracking with Kalman Filter
class FaceTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03
        self.last_measurement = None
        self.last_prediction = None
        
    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if self.last_measurement is None:
            self.last_measurement = measurement
            self.kalman.statePre = np.array([[measurement[0][0]], [measurement[1][0]], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[measurement[0][0]], [measurement[1][0]], [0], [0]], np.float32)
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        self.last_prediction = prediction
        return int(prediction[0][0]), int(prediction[1][0])

# Emotion detection labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Ensure necessary directories exist
os.makedirs('dataset', exist_ok=True)
os.makedirs('trainer', exist_ok=True)

# New: FPS counter
class FPSCounter:
    def __init__(self):
        self.start_time = None
        self.frame_count = 0
        self.fps = 0
        
    def update(self):
        if self.start_time is None:
            self.start_time = time.time()
        self.frame_count += 1
        if time.time() - self.start_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps

fps_counter = FPSCounter()

def clear_dataset():
    print("\nWarning: This will delete all collected data!")
    confirm = input("Are you sure you want to clear dataset? (yes/no): ").lower()
    
    if confirm == 'yes':
        try:
            for filename in os.listdir('dataset'):
                file_path = os.path.join('dataset', filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            
            if os.path.exists('trainer/face_recognition_model.h5'):
                os.remove('trainer/face_recognition_model.h5')
            if os.path.exists('trainer/class_names.npy'):
                os.remove('trainer/class_names.npy')
                
            print("Successfully cleared all dataset and trained models!")
        except Exception as e:
            print(f"Error while clearing dataset: {str(e)}")
    else:
        print("Operation cancelled.")

def collect_dataset():
    person_name = input("Enter person's name: ").strip()
    if not person_name:
        print("Error: Name cannot be empty!")
        return
        
    person_dir = os.path.join('dataset', person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    print(f"\nCollecting images for {person_name}.")
    print("Instructions:")
    print("1. Make sure only one face is visible")
    print("2. Press 's' to save image")
    print("3. Press 'q' to finish collection\n")
    
    count = len(os.listdir(person_dir))
    saving = False
    tracker = FaceTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access camera")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        face_img = None
        status = "No face detected"
        color = (0, 0, 255)  # Red
        
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            # Use Kalman filter for smoother tracking
            x, y = tracker.update(x + w//2, y + h//2)
            x, y = x - w//2, y - h//2
            
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            face_img = frame[y:y+h, x:x+w]
            status = "Face detected - Press 's' to save"
            color = (0, 255, 0)  # Green
        elif len(faces) > 1:
            status = "Multiple faces detected!"
            color = (0, 0, 255)  # Red
            
        # Display FPS
        fps = fps_counter.update()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Saved: {count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow('Face Data Collection', frame)
        
        key = cv2.waitKey(1)
        if key == ord('s') and len(faces) == 1:
            resized_face = cv2.resize(face_img, (100, 100))
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), resized_face)
            print(f"Saved image {count}.jpg")
            count += 1
            saving = True
        elif key == ord('q'):
            break
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return
    
    cv2.destroyAllWindows()
    if saving:
        print(f"\nSuccessfully collected {count} images of {person_name}")
    else:
        print("\nNo images were saved")

def load_dataset():
    X = []
    y = []
    class_names = []
    
    people = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
    
    if len(people) < 2:
        print("Error: You need at least 2 different people in dataset!")
        return None, None, None
    
    for label, person in enumerate(people):
        person_dir = os.path.join('dataset', person)
        class_names.append(person)
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (100, 100))
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                continue
    
    if len(X) == 0:
        print("Error: No valid images found in dataset!")
        return None, None, None
    
    X = np.array(X) / 255.0
    y = to_categorical(np.array(y), num_classes=len(people))
    
    return X, y, class_names

def train_model():
    X, y, class_names = load_dataset()
    if X is None:
        return
    
    print(f"\nFound {len(class_names)} people: {', '.join(class_names)}")
    print(f"Total images: {len(X)}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Enhanced data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    # Improved model architecture
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    print("\nModel Summary:")
    model.summary()
    
    batch_size = 32
    steps_per_epoch = max(1, len(X_train) // batch_size)
    validation_steps = max(1, len(X_val) // batch_size)
    
    print("\nStarting training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        epochs=25,
        verbose=1)
    
    model.save('trainer/face_recognition_model.h5')
    np.save('trainer/class_names.npy', np.array(class_names))
    
    print("\nTraining completed successfully!")
    print(f"Model can recognize {len(class_names)} people")

def recognize_faces():
    if not os.path.exists('trainer/face_recognition_model.h5'):
        print("\nError: Model not found!")
        print("Please train the model first using option 2")
        return
        
    model = load_model('trainer/face_recognition_model.h5')
    class_names = np.load('trainer/class_names.npy', allow_pickle=True)
    
    print("\nFace recognition started. Press ESC to exit")
    print(f"Recognizing {len(class_names)} people: {', '.join(class_names)}")
    
    # Load emotion detection model (simplified for demo)
    emotion_model = None
    try:
        emotion_model = load_model('trainer/emotion_model.h5')  # You would need to provide this
    except:
        print("Emotion model not found - skipping emotion detection")
    
    tracker = FaceTracker()
    emotion_history = {emotion: deque(maxlen=10) for emotion in EMOTIONS}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            # Use Kalman filter for smoother tracking
            x, y = tracker.update(x + w//2, y + h//2)
            x, y = x - w//2, y - h//2
            
            face_img = frame[y:y+h, x:x+w]
            resized = cv2.resize(face_img, (100,100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1,100,100,3))
            
            # Face recognition
            predictions = model.predict(reshaped, verbose=0)
            confidence = np.max(predictions)
            person_id = np.argmax(predictions)
            
            if confidence > 0.6:
                name = class_names[person_id]
                color = (0, 255, 0)  # Green
                label = f"{name} ({confidence*100:.1f}%)"
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red
                label = "Unknown"
            
            # Emotion detection
            emotion_label = ""
            if emotion_model:
                gray_face = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (48,48))
                gray_face = gray_face.reshape(1,48,48,1) / 255.0
                emotion_pred = emotion_model.predict(gray_face, verbose=0)[0]
                emotion_idx = np.argmax(emotion_pred)
                emotion = EMOTIONS[emotion_idx]
                emotion_history[emotion].append(emotion_pred[emotion_idx])
                
                # Get most common recent emotion
                avg_emotions = {e: np.mean(h) if h else 0 for e, h in emotion_history.items()}
                dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
                emotion_label = f"Emotion: {dominant_emotion}"
            
            # Draw face box and labels
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if emotion_label:
                cv2.putText(frame, emotion_label, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        # Display FPS
        fps = fps_counter.update()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()

def main():
    while True:
        print("\nFace Recognition System")
        print("1. Collect Dataset")
        print("2. Train Model")
        print("3. Recognize Faces")
        print("4. Clear Dataset")
        print("5. Exit")
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            collect_dataset()
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            clear_dataset()
        elif choice == '5':
            print("\nExiting program...")
            break
        else:
            print("\nInvalid choice. Please enter 1-5")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
