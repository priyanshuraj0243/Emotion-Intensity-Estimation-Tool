import numpy as np
import cv2
import os
import sys
import time
import matplotlib.pyplot as plt
from threading import Thread
from playsound import playsound
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import tempfile

# --- Handle Working Directory for PyInstaller vs Source ---
if getattr(sys, 'frozen', False):
    # If running as a compiled .exe, use the temp directory
    os.chdir(tempfile.gettempdir())
else:
    # If running from source, set working directory to the project root 
    # (one level up from the 'src' folder where this script lives)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

# TensorFlow suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model

# --- Helper for PyInstaller paths ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Load model and haarcascade (Updated for repo structure) ---
model_path = resource_path(os.path.join('models', 'best_regression_model.h5'))
cascade_path = resource_path(os.path.join('models', 'haarcascade_frontalface_default.xml'))
alert_sound_path = resource_path(os.path.join('src', 'alert.mp3'))

model = load_model(model_path)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cascade_path)

current_emotions = [0]*7
last_log_time = time.time()

def play_alert():
    try:
        playsound(alert_sound_path)
    except:
        pass

def log_mood(top_emotion):
    # Ensure the logs directory exists before writing
    os.makedirs('logs', exist_ok=True)
    log_file_path = os.path.join('logs', 'mood_log.csv')
    
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), top_emotion])

def live_pie_chart():
    plt.ion()
    fig, ax = plt.subplots(figsize=(4, 4))
    while True:
        ax.clear()
        ax.pie(current_emotions, labels=emotion_labels, autopct='%1.1f%%', colors=plt.cm.Dark2.colors)
        ax.set_title("Live Emotion Analysis")
        plt.pause(1)

def predict_emotion(frame):
    global current_emotions, last_log_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face, verbose=0)[0]
        preds_percent = preds * 100
        current_emotions = preds_percent

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sorted_emotions = dict(sorted({emotion_labels[i]: preds_percent[i] for i in range(len(emotion_labels))}.items(), key=lambda item: item[1], reverse=True))
        for idx, (emotion, percent) in enumerate(list(sorted_emotions.items())[:3]):
            text = f"{emotion}: {percent:.1f}%"
            text_y = y + 20 + idx*20
            if text_y > y + h - 10:
                text_y = y + h - 10 - (2-idx)*20
            cv2.putText(frame, text, (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if preds_percent[0] > 70 or preds_percent[4] > 70:
            Thread(target=play_alert).start()
            cv2.putText(frame, '!! ALERT: High Negative Emotion !!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if time.time() - last_log_time > 10:
            top_emotion = max(sorted_emotions, key=sorted_emotions.get)
            log_mood(top_emotion)
            last_log_time = time.time()

    return frame

def real_time_emotion_regression():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    Thread(target=live_pie_chart, daemon=True).start()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = predict_emotion(frame)
        current_time = time.time()
        fps = int(1/(current_time - prev_time))
        prev_time = current_time
        cv2.putText(frame, f'FPS: {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, 'Built by Priyanshu Raj', (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Webcam Emotion Regression - PRIYANSHU RAJ', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Webcam Emotion Regression - PRIYANSHU RAJ', 0) < 0:
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')

def upload_image():
    filepath = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if filepath:
        img = cv2.imread(filepath)
        frame = predict_emotion(img)
        try:
            cv2.imshow('Image Emotion Prediction - PRIYANSHU RAJ', frame)
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Image Emotion Prediction - PRIYANSHU RAJ', 0) < 0:
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to show image: {e}")

def exit_app():
    root.destroy()

# --- GUI ---
root = tk.Tk()
root.title("Emotion Regression Tool - PRIYANSHU RAJ")
root.geometry("400x300")
root.configure(bg='#1e1e1e')

title_label = tk.Label(root, text="Emotion Regression Tool", font=("Helvetica", 16, 'bold'), bg='#1e1e1e', fg='white')
title_label.pack(pady=20)

btn_camera = tk.Button(root, text="Use Camera", font=("Helvetica", 14), command=real_time_emotion_regression, bg='#2e2e2e', fg='white')
btn_camera.pack(pady=10)

btn_upload = tk.Button(root, text="Upload Image", font=("Helvetica", 14), command=upload_image, bg='#2e2e2e', fg='white')
btn_upload.pack(pady=10)

btn_exit = tk.Button(root, text="Exit", font=("Helvetica", 14), command=exit_app, bg='#2e2e2e', fg='white')
btn_exit.pack(pady=10)

footer_label = tk.Label(root, text="Built by PRIYANSHU RAJ", font=("Helvetica", 10), bg='#1e1e1e', fg='white')
footer_label.pack(side="bottom", pady=10)

root.mainloop()