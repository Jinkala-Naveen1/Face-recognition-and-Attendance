import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

# Directory containing pre-trained images
trained_images_dir = 'ImageBasics'

# Load and encode pre-trained images
def load_pretrained_images(directory):
    known_encodings = []
    known_names = []
    
    for filename in os.listdir(directory):
        if filename.endswith(('.jpeg', '.jpg', '.png')):
            img_path = os.path.join(directory, filename)
            img = face_recognition.load_image_file(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Encode the image
            encodings = face_recognition.face_encodings(img_rgb)
            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)
    
    return known_encodings, known_names

# Store encodings of pre-trained images
known_encodings, known_names = load_pretrained_images(trained_images_dir)

# Load and preprocess the new image
def process_new_image(image_path):
    img = face_recognition.load_image_file(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(img_rgb)
    return encodings[0] if encodings else None

# Compare new image with pre-trained images and log every recognition
def check_new_image_and_log_attendance(new_image_encoding, known_encodings, known_names, attendance_log):
    results = face_recognition.compare_faces(known_encodings, new_image_encoding)
    face_distances = face_recognition.face_distance(known_encodings, new_image_encoding)
    
    if True in results:
        match_index = results.index(True)
        matched_name = known_names[match_index]
        confidence = (1 - face_distances[match_index]) * 100
        
        # Log attendance every time a match is found
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        attendance_log.append([matched_name, current_time, confidence])
        return matched_name, confidence
    else:
        return None, None

# Initialize attendance log
attendance_log = []

# Example usage with a new image
new_image_path = 'ImageBasics/Elon_musk_test.jpeg'
new_image_encoding = process_new_image(new_image_path)

if new_image_encoding:
    matched_name, confidence = check_new_image_and_log_attendance(new_image_encoding, known_encodings, known_names, attendance_log)
    
    if matched_name:
        print(f"Match found: {matched_name} with {confidence:.2f}% confidence")
        print(f"Attendance logged for {matched_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("No match found. New image detected.")
else:
    print("No face detected in the new image.")

# Convert attendance log to DataFrame
if attendance_log:
    df_attendance = pd.DataFrame(attendance_log, columns=['Name', 'Timestamp', 'Confidence'])
    df_attendance.to_csv('attendance_log.csv', index=False)
    print("Attendance log saved to 'attendance_log.csv'")
else:
    print("No attendance logged.")
