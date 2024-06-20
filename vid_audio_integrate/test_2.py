import cv2
import numpy as np
import os
import dlib
from gtts import gTTS
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
from scipy.spatial import ConvexHull

# Path to dlib's pre-trained model for facial landmark detection
PREDICTOR_PATH = "face_recognition_densenet_model_v1.dat"

# Function to convert text to speech and save as an audio file
def text_to_speech(text, audio_file):
    tts = gTTS(text=text, lang='en')
    tts.save(audio_file)
    os.chmod(audio_file, 0o777)  # Change permissions to 777
    print(f"Audio file saved as {audio_file}")

# Function to detect facial landmarks
def get_landmarks(image, detector, predictor):
    rects = detector(image, 1)
    if len(rects) > 0:
        return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])
    else:
        return None

# Function to draw the mouth
def draw_mouth(image, landmarks, mouth_points):
    hull = ConvexHull(landmarks[mouth_points])
    mouth_hull = landmarks[mouth_points][hull.vertices]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, mouth_hull, 255)
    image[mask == 255] = [0, 0, 255]  # Mouth color
    return image

# Function to create animated video from a photo
def create_animated_video(image_path, audio_file, video_file):
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file {audio_file} not found.")
    
    # Load audio file to get its duration
    try:
        audio = AudioSegment.from_file(audio_file, format="mp3")
    except Exception as e:
        raise Exception(f"Error loading audio file: {e}")
    
    audio_duration = len(audio) / 1000.0  # duration in seconds

    # Video settings
    width, height = 640, 480
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))

    # Initialize dlib's face detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # Detect landmarks
    landmarks = get_landmarks(image, detector, predictor)
    if landmarks is None:
        raise Exception("No face detected in the image.")
    
    # Mouth points based on dlib's 68-point model
    mouth_points = list(range(48, 61))

    num_frames = int(audio_duration * fps)
    for i in range(num_frames):
        frame = image.copy()
        frame = draw_mouth(frame, landmarks, mouth_points)
        out.write(frame)

    out.release()
    print(f"Video file saved as {video_file}")

# Function to combine the photo video and audio
def combine_audio_video(audio_file, video_file, output_file):
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file {video_file} not found.")
    
    video_clip = VideoFileClip(video_file)
    audio_clip = AudioFileClip(audio_file)
    video = video_clip.set_audio(audio_clip)
    video.write_videofile(output_file, codec='libx264')
    print(f"Final video saved as {output_file}")

if __name__ == "__main__":
    text = "Hello, this is a sample text to demonstrate text to video conversion."
    audio_file = "output_audio_2.mp3"
    image_path = "04.jpg"  # Replace with your photo path
    video_file = "output_video_2.avi"
    output_file = "final_video_2.mp4"

    # Convert text to speech
    text_to_speech(text, audio_file)
    
    # Create animated video from photo
    create_animated_video(image_path, audio_file, video_file)
    
    # Combine audio and video
    combine_audio_video(audio_file, video_file, output_file)

    print("Video generation complete.")
