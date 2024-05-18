# YouTube Video Summarizer and Q&A

This Streamlit web application allows you to extract the transcript from a YouTube video, generate a summary of its content, and ask questions about the video's content.

## Features

- **Transcript Extraction:** Download the audio from a YouTube video and transcribe it into text.
- **Content Summarization:** Automatically generate a summary of the video's transcript.
- **Question & Answer:** Ask questions about the video's content and receive answers based on the generated transcript.

## Screenshots

![Screenshot of the application](screenshot.png)

## Libraries Used

- **Streamlit:** Streamlit is used for building and deploying interactive web applications with Python.
- **pytube:** pytube is a library for downloading YouTube videos.
- **Whisper:** Whisper is used for audio transcription.
- **transformers:** The transformers library from Hugging Face is used for natural language processing tasks such as summarization and question answering.

## Usage

1. **Enter YouTube Video URL:** Paste the URL of the YouTube video you want to analyze.
2. **Process Video:** Click on the "Process Video" button to initiate the analysis.

## Installation

To run this application locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your-repo
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run your_app_name.py
   ```

## Models Used

- **Transcription Model:** This application uses the `Whisper` library for audio transcription. The model employed is the "base" model provided by Whisper.
- **Summarization Model:** The summarization functionality utilizes the `Facebook BART (BART-large-CNN)` model from Hugging Face Transformers. It generates concise summaries of the video transcript.
- **Question Answering Model:** For answering user questions about the video content, the application utilizes the `deepset Roberta-based SQuAD2` model from Hugging Face Transformers.

## Code Overview

- **YouTube Video Processing:** The code extracts the video ID from the YouTube URL and downloads the audio from the video using the pytube library.
- **Transcription:** Audio transcription is performed using the Whisper library. The generated transcript is then chunked into smaller segments.
- **Summarization:** The application utilizes the Hugging Face `transformers` library to summarize the transcript chunks using the BART model.
- **Question Answering:** User questions are answered based on the generated transcript using the `Roberta-based SQuAD2` model.
