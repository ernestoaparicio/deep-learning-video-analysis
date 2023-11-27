import cv2
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pytesseract
from PIL import Image
from googleapiclient.discovery import build
from textblob import TextBlob
from googleapiclient.discovery import build
from textblob import TextBlob
from emotion_recognition_model import get_base_model
from emotion_recognition_utils import preprocess_fer, get_labels_fer
from pytube import YouTube

import json


# Initialize the YouTube API client
youtube = build('youtube', 'v3', developerKey='AIzaSyD1FMUZtNeJdcdD9ddmyoV0UMDDFuuTqrM')

# Facial Emotion Recognition Model Setup
facial_recognition_model = get_base_model((100, 100, 3))
facial_recognition_model.add(tf.keras.layers.Dense(7, activation='softmax', name="softmax"))

facial_recognition_model_name = 'FERplus_0124-1040_weights.h5'
facial_recognition_model.load_weights('./models/' + facial_recognition_model_name)


# Load the pre-trained SSD MobileNet model from TensorFlow Hub for object detection
object_detection_model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1'
object_detection_model = hub.load(object_detection_model_handle).signatures['serving_default']

def get_categorized_comments(video_id):
    # Fetch and categorize comments for each video
    return get_video_comments_with_categories(video_id)
    

# Function to get video metadata
def get_video_metadata(video_id):
    request = youtube.videos().list(part="snippet,contentDetails", id=video_id)
    response = request.execute()

    if response['items']:
        return response['items'][0]
    else:
        return None

def download_youtube_video(video_id):
    filename = f'{video_id}.mp4'

    # Check if the file already exists
    if not os.path.exists(filename):
        yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
        video_stream = yt.streams.filter(file_extension='mp4').first()
        video_stream.download(filename=filename)
    else:
        print(f"{filename} already exists. Skipping processing.")

    return filename

def classify_frame(instructor_present, instructor_bbox, text_detected):
    if instructor_present and text_detected:
        return 'blackboard'
    elif text_detected:
        return 'powerpoint'
    else:
        return 'neither'

# Function to detect text in the frame using Tesseract OCR
def detect_text(frame):
    # Convert the frame to a PIL Image
    pil_img = Image.fromarray(frame)
    # Use pytesseract to get bounding boxes of text
    boxes = pytesseract.image_to_boxes(pil_img)

    # Calculate the total area covered by text
    text_area = 0
    for b in boxes.splitlines():
        b = b.split(' ')
        text_area += (int(b[3]) - int(b[1])) * (int(b[4]) - int(b[2]))

    frame_area = frame.shape[0] * frame.shape[1]
    text_density = text_area / frame_area if frame_area else 0

    return text_density

def detect_instructor_and_text(frame, presence_threshold=0.5):
    # Detect if an instructor is present in the frame
    instructor_present, instructor_bbox = detect_instructor(frame, presence_threshold)

    # Detect text in the frame using Tesseract OCR
    text_detected = detect_text(frame)

    return instructor_present, instructor_bbox, text_detected

def get_video_comments(video_id):
    # Fetch comments from the video
    comments = []
    response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100  # You can adjust this
    ).execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            # Check and fetch replies if available
            if 'replies' in item and item['snippet']['totalReplyCount'] > 0:
                for reply_item in item['replies']['comments']:
                    reply = reply_item['snippet']['textDisplay']
                    comments.append(reply)

        # Check if there are more comments
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=response['nextPageToken'],
                textFormat='plainText',
                maxResults=100
            ).execute()
        else:
            break

    return comments

def categorize_comment(comment):
    # Analyze the sentiment of the comment
    analysis = TextBlob(comment)
    # Check if it's a question
    if comment.endswith('?'):
        return 'question'
    # Categorize based on sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive comment'
    elif analysis.sentiment.polarity == 0:
        return 'neutral comment'
    else:
        return 'negative comment'

def get_video_comments_with_categories(video_id):
    # Fetch comments from the video
    categorized_comments = []
    response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        textFormat='plainText',
        maxResults=100  # You can adjust this
    ).execute()

    while response:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            category = categorize_comment(comment)
            categorized_comments.append((comment, category))
            # Check and fetch replies if available
            if 'replies' in item and item['snippet']['totalReplyCount'] > 0:
                for reply_item in item['replies']['comments']:
                    reply = reply_item['snippet']['textDisplay']
                    reply_category = categorize_comment(reply)
                    categorized_comments.append((reply, reply_category))

        # Check if there are more comments
        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=response['nextPageToken'],
                textFormat='plainText',
                maxResults=100
            ).execute()
        else:
            break

    return categorized_comments
    

def analyze_video_for_text(video_folder):
    blackboard, powerpoint, neither, total_frames, avg_text_density = process_video_segments(video_folder)

    if total_frames > 0:
        fraction_blackboard = blackboard / total_frames
        fraction_powerpoint = powerpoint / total_frames
        fraction_neither = neither / total_frames

        print(f"Fraction of Blackboard in {video_folder}: {fraction_blackboard}")
        print(f"Fraction of PowerPoint in {video_folder}: {fraction_powerpoint}")
        print(f"Fraction of Neither in {video_folder}: {fraction_neither}")
        print(f"Average Text Density in each frame {video_folder}: {avg_text_density}")
    else:
        print(f"No frames to analyze in {video_folder}")

def analyze_facial_expression(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = cv2.resize(img, dsize=IMG_SHAPE[:-1])
    x = np.expand_dims(x, axis=0)
    x = preprocess_fer(x)

    output = model.predict(x)
    label = get_labels_fer(output)[0]
    confidence = np.argmax(output[0])

    return label, confidence

def process_keyframes(keyframe_folder):
    expression_counts = {emotion: 0 for emotion in ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']}
    total_frames = 0

    for keyframe_file in os.listdir(keyframe_folder):
        keyframe_path = os.path.join(keyframe_folder, keyframe_file)
        keyframe = cv2.imread(keyframe_path)
        
        if keyframe is not None:
            total_frames += 1
            label, _ = analyze_facial_expression(keyframe)
            expression_counts[label] += 1

    # Calculate expression percentages
    if total_frames > 0:
        for expression in expression_counts:
            expression_counts[expression] = expression_counts[expression] / total_frames

    return expression_counts

def analyze_videos(video_folders):
    for video_folder in video_folders:
        print(f"Analyzing keyframes in {video_folder}")
        expression_percentages = process_keyframes(video_folder)
        print(f"Facial Expression Percentages for {video_folder}:")
        for expression, percentage in expression_percentages.items():
            print(f"{expression.capitalize()}: {percentage * 100:.2f}%")
        print(f"Finished analyzing {video_folder}")

def detect_instructor(frame, presence_threshold=0.5):
    # Convert the frame to uint8 and process for model input
    frame = tf.image.convert_image_dtype(frame, tf.uint8)
    input_tensor = tf.expand_dims(frame, 0)

    # Model inference
    result = object_detection_model(input_tensor)

    # Parse the results
    result = {key:value.numpy() for key,value in result.items()}
    detection_scores = result["detection_scores"]
    detection_classes = result["detection_classes"]
    detection_boxes = result["detection_boxes"]

    # Check for instructor presence (class 1 is 'person')
    for score, clss, box in zip(detection_scores[0], detection_classes[0], detection_boxes[0]):
        if score >= presence_threshold and clss == 1:
            return True, box
    return False, None

def is_full_screen(bbox, frame_shape):
    frame_height, frame_width, _ = frame_shape
    bbox_width = bbox[3] - bbox[1]  # Calculate width of bounding box
    bbox_height = bbox[2] - bbox[0]  # Calculate height of bounding box

    # Example thresholds: 50% of frame width and 50% of frame height for full-screen
    min_full_screen_width = frame_width * 0.5
    min_full_screen_height = frame_height * 0.5

    if bbox_width >= min_full_screen_width and bbox_height >= min_full_screen_height:
        return True
    return False

def process_video_segments(segment_folder):
    instructor_presence_count = 0
    total_frames = 0
    full_screen_count = 0
    pip_count = 0

    for segment_file in os.listdir(segment_folder):
        segment_path = os.path.join(segment_folder, segment_file)
        frame = cv2.imread(segment_path)
        if frame is not None:
            total_frames += 1
            present, bbox = detect_instructor(frame)
            if present:
                instructor_presence_count += 1
                if is_full_screen(bbox, frame.shape):
                    full_screen_count += 1
                else:
                    pip_count += 1

    return instructor_presence_count, total_frames, full_screen_count, pip_count

def analyze_video_for_instructor(video_folder, sampling_rate):
    instructor_presence, total_frames, full_screen, pip = process_video_segments(video_folder)

    if total_frames > 0:
        fraction_visible = instructor_presence / total_frames
        fraction_full_screen = full_screen / total_frames
        fraction_pip = pip / total_frames

        # Calculate total time in seconds
        total_time_seconds = instructor_presence * sampling_rate
        # Convert to mm:ss format
        minutes, seconds = divmod(total_time_seconds, 60)
        total_time_present = f"{int(minutes):02d}:{int(seconds):02d}"
    else:
        total_time_present = "00:00"

    return {
        'instructor_presence': instructor_presence,
        'fraction_visible': fraction_visible,
        'sampling_rate': sampling_rate,
        'total_time_present': total_time_present
    }


def process_video_for_keyframes(video_path, sampling_rate):
    sampling_rate = sampling_rate  # sampling_rate in seconds
    cap = cv2.VideoCapture(video_path)
    keyframe_folder_new = True

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get frame rate
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Calculate sampling interval in frames
    sampling_interval = int(frame_rate * sampling_rate)  # sampling_rate in seconds

    frame_count = 0
    keyframe_count = 0

    # Folder for keyframes
    base_folder = os.path.splitext(os.path.basename(video_path))[0]
    keyframe_folder = f'keyframes/{base_folder}'

    # Check if the keyframe folder already exists
    if os.path.exists(keyframe_folder):
        print(f"Keyframe folder {keyframe_folder} already exists. Skipping processing.")
        keyframe_folder_new = True

    if keyframe_folder_new:
        os.makedirs(keyframe_folder, exist_ok=True)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sampling_interval == 0:
                # Save keyframe
                keyframe_file = f'{keyframe_folder}/keyframe_{keyframe_count}.jpg'
                cv2.imwrite(keyframe_file, frame)
                keyframe_count += 1

            frame_count += 1
        cap.release()

    keyframe_data = {
        'frame_count': frame_count,
        'keyframe_count': keyframe_count,
        'frame_rate': frame_rate,
        'sampling_interval': sampling_interval,
        'sample_rate': sampling_rate
    }
    return keyframe_data



# Main function to analyze a YouTube video
def analyze_youtube_video(video_id):
    # Step 1: Download the video (implement this part as per your method)
    video_path = download_youtube_video(video_id)  # This function needs to be implemented

    # Step 2: Extract metadata (using YouTube API)
    metadata = get_video_metadata(video_id)  # Implement this function using YouTube API

    # Step 3: Process the video for keyframes and other analyses
    keyframe_data = process_video_for_keyframes(video_path, sampling_rate=1)
    instructor_presence_data = analyze_video_for_instructor('keyframes/' + video_id, 1)

    # Step 4: Fetch and categorize comments
    categorized_comments = get_categorized_comments(video_id)  # Implement using YouTube API and TextBlob

    # Step 5: Compile all results into a structured format
    results = {
        'Playlist ID': metadata['snippet'].get('channelTitle', 'NA'),
        'Video ID': metadata.get('id', 'NA'),
        'Video Title': metadata['snippet'].get('title', 'NA'),
        'Total Duration': metadata['contentDetails'].get('duration', 'NA'),
        'Number of Segments': keyframe_data.get('frame_count', 'NA'),
        'Number of Keyframes': keyframe_data.get('keyframe_count', 'NA'),
        'Timing of each keyframe': keyframe_data.get('sampling_rate', 'NA'),
        'Instructor Presence': instructor_presence_data.get('fraction_visible', 'NA'),
        'The total time when the instructor is present (mm:ss)': instructor_presence_data.get('total_time_present', 'NA'),
        'Body movement (Y/N)':'NA',
        'Use of Slides (mm:ss)':'NA', 
        'Use of Blackboard (mm:ss)': 'NA',
        'Average fraction of text on slides':'NA',
        'total slide area':'NA',
        'Number of Total Comments':'NA', 
        'Number of Positive comments':'NA', 
        'Number of Negative comments':'NA', 
        'Number of questions':'NA'
    }
    return results

# Example usage
video_id = 'BRMS3T11Cdw'
results = analyze_youtube_video(video_id)
print(results)
