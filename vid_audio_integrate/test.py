import cv2
import numpy as np
import os
from gtts import gTTS
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

# Function to convert text to speech and save as an audio file
def text_to_speech(text, audio_file):
    tts = gTTS(text=text, lang='en')
    tts.save(audio_file)
    os.chmod(audio_file, 0o777)
    print(f"Audio file saved as {audio_file}")

# Function to create a basic animated avatar video
def create_avatar_video(audio_file, video_file):
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

    # Avatar settings
    avatar_color = (0, 255, 0)
    mouth_color = (0, 0, 255)
    mouth_open_height = 20
    mouth_closed_height = 5

    num_frames = int(audio_duration * fps)
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw avatar's head
        cv2.circle(frame, (width // 2, height // 2), 100, avatar_color, -1)
        
        # Simulate mouth movement
        mouth_height = mouth_open_height if (i // fps) % 2 == 0 else mouth_closed_height
        cv2.rectangle(frame, (width // 2 - 50, height // 2 + 50),
                      (width // 2 + 50, height // 2 + 50 + mouth_height), mouth_color, -1)
        
        out.write(frame)

    out.release()
    print(f"Video file saved as {video_file}")

# Function to combine the avatar video and audio
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
    audio_file = "output_audio.mp3"
    video_file = "output_video.avi"
    output_file = "final_video.mp4"

    # Convert text to speech
    text_to_speech(text, audio_file)
    
    # Create avatar video
    create_avatar_video(audio_file, video_file)
    
    # Combine audio and video
    combine_audio_video(audio_file, video_file, output_file)

    print("Video generation complete.")
