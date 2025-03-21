import streamlit as st 
from google import genai
from google.genai import types
import speech_recognition as sr 
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import os
from io import BytesIO
from PIL import Image
# for image to video
import cv2 as cv
import imageio as iio
import numpy as np
# for audio duration
from pydub import AudioSegment
import glob

# Set up Gemini API client
client = genai.Client(api_key=st.secrets["gemini"]["api_key"])

# Define Directories
images_dir = "Images"
os.makedirs(images_dir, exist_ok=True)  # Ensure directory exists
# Define the audio directory path
audio_path = r"Audio/generated_audio"
os.makedirs(audio_path, exist_ok=True)  # Ensure directory exists
# Define the text files directory path
text_file_path = r"Text_files"
os.makedirs(text_file_path, exist_ok=True)  # Ensure directory exists

# Initialize session state variables
if "prev_input" not in st.session_state:
    st.session_state.prev_input = ""  # Stores the last input question
if "response_text" not in st.session_state:
    st.session_state.response_text = None  # Stores AI-generated response
if "response_prompt" not in st.session_state:
    st.session_state.response_prompt = None  # Stores AI-generated image prompt response
if "speech_generated" not in st.session_state:
    st.session_state.speech_generated = False  # Tracks if speech was generated
if "audio_files" not in st.session_state:
    st.session_state.audio_files = []  # List of generated audio files
if "image_files" not in st.session_state:
    st.session_state.image_files = []  # List of generated image files

# Function to delete old files
def delete_files_in_directory(directory_path):
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError as e:
        print(f"Error: {e}")


# User input
user_input = st.text_input("Write to Generate")

# Reset response & delete old audio files if the user input changes
if user_input != st.session_state.prev_input:
    st.session_state.response_text = None
    st.session_state.response_prompt = None
    st.session_state.speech_generated = False
    st.session_state.prev_input = user_input  # Update stored input
    delete_files_in_directory(audio_path)  # Delete old audio files
    delete_files_in_directory(images_dir)  # Delete old images

# Generate response if input is given and response is not already stored
if user_input and st.session_state.response_text is None:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_input,
    )
    st.session_state.response_text = response.text  # Store response
    
# Generate second response for image prompt
if st.session_state.response_text and st.session_state.response_prompt is None:
    response_prompt = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=st.session_state.response_text + """(Generate an image prompt for each individual sentence in the provided story, ensuring that the total number of prompts exactly matches the number of sentences ending with a full stop (period). When a sentence ends with a full stop, generate a new, distinct image prompt for that specific sentence only. This means that if the story contains ten sentences ending in a full stop, you will generate ten image prompts.
        Each prompt should vividly and realistically (not anime style) depict the specific moment of its corresponding sentence, maintaining a natural progression. Write each prompt as a separate paragraph without any labels or headings. Focus on rich details, character expressions, atmosphere, and relevant elements to create visually compelling and cohesive imagery that aligns with the narrative. Ensure continuity between the images, depicting the same characters (if present) and maintaining a consistent visual style to create a cohesive sequence of illustrations. The goal is to create a seamless visual storytelling experience, with each prompt capturing a specific moment in the story in high detail.)""",
    )
    st.session_state.response_prompt = response_prompt.text  # Store second response

# Display first response
if st.session_state.response_text:
    st.write(st.session_state.response_text)

    # Save response to file
    with open(os.path.join(text_file_path,"story.txt"), "w") as story_file:
        story_file.write(st.session_state.response_text)

# Display prompt response     
if st.session_state.response_prompt:
    st.write("**Generated Image Prompt:**")
    st.write(st.session_state.response_prompt)

    # Save response to file
    with open(os.path.join(text_file_path,"images_prompts.txt"), "w") as prompt_file:
        prompt_file.write(st.session_state.response_prompt)



# Load the TTS model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load xvector for speaker voice characteristics
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Read the stored response text
with open(os.path.join(text_file_path,"story.txt"), "r") as file:
    story_text = file.read()

# Split text into sentences/paragraphs
sentences = story_text.split(". ")  # Modify if needed

# Generate speech for all sentences at once
if st.button("Generate All Voices"):
    if sentences and not st.session_state.speech_generated:
        delete_files_in_directory(audio_path)  # Ensure no old files exist before generating new ones
        
        audio_files = []
        for idx, sentence in enumerate(sentences):
            if sentence.strip():  # Ignore empty sentences
                # Generate speech
                inputs = processor(text=sentence, return_tensors="pt")
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

                # Save each sentence as a separate audio file
                filename = os.path.join(audio_path, f"speech_{idx + 1}.wav")
                sf.write(filename, speech.numpy(), samplerate=16000)
                audio_files.append(filename)

        # Store generated state
        st.session_state.speech_generated = True
        st.session_state.audio_files = audio_files

# Display all generated audio files
if st.session_state.speech_generated:
    st.write("Generated Speech Files:")
    for filename in st.session_state.audio_files:
        st.audio(filename, format="audio/wav")  # Play audio in Streamlit
        st.markdown(f'<a href="{filename}" download>Download {os.path.basename(filename)}</a>', unsafe_allow_html=True)


if st.button("Generate Images"):
    
    # Read prompts and filter out empty lines
    with open(os.path.join(text_file_path,"images_prompts.txt"), "r") as file:
        contents = [line.strip() for line in file.readlines() if line.strip()]
        
    # Delete old images
    delete_files_in_directory(images_dir)

    # Generate images for each prompt
    for idx, content in enumerate(contents):
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=content,  # Pass one prompt at a time
            config=types.GenerateContentConfig(response_modalities=["Text", "Image"])
        )
        st.write(f"# Prompt Image {idx + 1}")
        st.write(contents[idx])
        st.write(f"### Generated Image {idx + 1}")  # Label each image

        # Iterate over response parts
        for part in response.candidates[0].content.parts:
            try:
                if part.text:
                    st.write("Text Response:", part.text)  # Display any text response
                elif part.inline_data:
                    # Debugging: Check the content type
                    st.write("MIME Type:", part.inline_data.mime_type)
                    st.write("Data Size:", len(part.inline_data.data))

                    if "image" in part.inline_data.mime_type:  # Ensure it's an image
                        try:
                            image_data = BytesIO(part.inline_data.data)  # Remove base64 decoding
                            image = Image.open(image_data)

                            # Save image
                            image_path = os.path.join(images_dir, f"image_{idx + 1}.png")
                            image.save(image_path)

                            # Display image in Streamlit
                            st.image(image_path, caption=f"Image {idx + 1}")

                        except Exception as e:
                            st.error(f"Error loading image {idx + 1}: {e}")
                    else:
                        st.error("The response does not contain an image.")
            except Exception as e:
                st.error(f"Error generating image for prompt {idx+1}: {e}") 
                
                


if st.button("Generate Video"):

  

    # Define the list of audio files
    audio_path = r"Audio/generated_audio/"
    os.makedirs(audio_path, exist_ok=True)  # Ensure directory exists
    audios = [audio for audio in os.listdir(audio_path) if audio.endswith(".wav")]
    audios = [audio_path + audio for audio in audios]

    # Alternatively, use glob to find all audio files in a directory
    # audio_files = glob.glob('path/to/audio/*.mp3')

    # Initialize an empty list to store the durations
    durations = []

    # Iterate through the list of audio files
    for file in audios:
        # Load the audio file
        audio = AudioSegment.from_file(file)
        # Get the duration in seconds
        duration_secs = audio.duration_seconds
        # duration_secs = int(audio.duration_seconds)
        # Append the duration to the list
        durations.append(duration_secs)

    # Print the list of durations in milliseconds
    st.write(durations)




    # Path to the images
    image_path = r"Images"
    os.makedirs(image_path, exist_ok=True)  # Ensure directory exists
    delete_files_in_directory("Video")
    # Output video file path
    output_video = "Video/output_video.mp4"
    # List all .png images in the directory
    images = [img for img in os.listdir(image_path) if img.endswith(".png")]

    # Prepend the image path to each image file name
    images = [os.path.join(image_path, img) for img in images]

    # Display the list of image paths
    st.write(images)

    # Read the first image to get its dimensions
    image = cv.imread(images[0])
    height, width, _ = image.shape

    # Read images and create a list of image arrays, resizing them to the first image's dimensions
    images = [cv.resize(cv.imread(image_file), (width, height)) for image_file in images]


    # Duration of each image in the video (in seconds)
    frame_duration = durations # Each image will be displayed according to list
    transition_duration = 0.4  # Duration of the transition in seconds
    fps = 24  # Frames per second

    # Function to create a fade transition between two images
    def create_fade_transition(image1, image2, duration, fps):
        num_frames = int(duration * fps)
        transition_frames = []
        for i in range(num_frames):
            alpha = i / num_frames
            blended = cv.addWeighted(image1, 1 - alpha, image2, alpha, 0)
            transition_frames.append(blended)
        return transition_frames

    # Create a video writer object
    with iio.get_writer(output_video, fps=fps) as writer:
        for i in range(len(images)):
            # Convert the image from BGR to RGB
            image_rgb = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
            
            # Write the image frames
            for _ in range(int(frame_duration[i] * fps)):
                writer.append_data(image_rgb)
            
            # Create and write transition frames if not the last image
            if i < len(images) - 1:
                transition_frames = create_fade_transition(images[i], images[i + 1], transition_duration, fps)
                for frame in transition_frames:
                    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    writer.append_data(frame_rgb)

    st.write(f"Video saved as {output_video}")
    st.video(output_video)


