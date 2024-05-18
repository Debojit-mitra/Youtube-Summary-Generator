#1. Youtube video summarizer and Q&A based on wisper and I have used transformer pipeline models rather than using gpt and its annoying limited key
#2. Make sure you have ffmpeg is there and added to %path%
#3. It downloads models like the wisper-base model, and other two models for summary and question answer
#4. Larger the video the more time will be required by wisper to generate transcript because model will be run in local machine
#5. (ImportError: cannot import name 'pipeline' from 'transformers') <- if this error occurs just refresh page

import streamlit as st
from pytube import YouTube
import whisper
import os
from transformers import pipeline

# Function to extract video ID from YouTube URL
def get_video_id(url):
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    return None

# Function to download YouTube video and extract audio
def download_youtube_video(video_id):
    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
        stream = yt.streams.filter(only_audio=True).first()
        out_file = stream.download(output_path=".")
        base, ext = os.path.splitext(out_file)
        new_file = base + '.mp3'
        os.rename(out_file, new_file)
        return new_file
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result['text']
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# Function to chunk text into smaller segments
def chunk_text(text, chunk_size=512):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to validate if the question is relevant to the video content
def is_relevant_question(question, transcript):
    keywords = set(question.lower().split())
    transcript_words = set(transcript.lower().split())
    common_keywords = keywords.intersection(transcript_words)
    return len(common_keywords) > 0

# Initialize summarization and QA pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Streamlit UI
st.title("YouTube Video Summarizer and Q&A")
youtube_url = st.text_input("Enter YouTube video URL:")

# Initialize session state variables
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

# Add "Process Video" button
process_button = st.button("Process Video")

# Process video if URL is provided and the button is pressed
if youtube_url and process_button:
    video_id = get_video_id(youtube_url)
    if video_id:
        download_message = st.empty()
        download_message.text("Downloading video and extracting audio...")
        audio_file = download_youtube_video(video_id)
        if audio_file and os.path.exists(audio_file):
            download_message.text("Transcribing audio...")
            transcript = transcribe_audio(audio_file)
            os.remove(audio_file)  # Clean up downloaded file
            if transcript:
                st.session_state.transcript = transcript
                download_message.text("Transcript generated successfully.")
                
                download_message.text("Generating summary...")
                chunks = chunk_text(transcript)
                summary = ' '.join([summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks])
                st.session_state.summary = summary
                download_message.text("")
        else:
            st.error("Failed to process the audio file.")
    else:
        st.error("Invalid YouTube URL")

# Display transcript and summary if available
if st.session_state.transcript:
    with st.expander("Show Transcript"):
        st.write(st.session_state.transcript)
    if st.session_state.summary:
        st.write("Summary:")
        st.write(st.session_state.summary)

# Ask a question about the video content
if st.session_state.transcript:
    st.write("Ask a question about the video content:")
    question = st.text_input("Enter your question:")
    if question:
        if is_relevant_question(question, st.session_state.transcript):
            result = qa_pipeline(question=question, context=st.session_state.transcript)
            st.write("Answer:")
            st.write(result['answer'])
        else:
            st.error("The question does not seem to be related to the video content. Please ask a relevant question.")




