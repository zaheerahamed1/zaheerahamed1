
import os
import openai
import logging
import whisper
import re

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize Whisper model (using base for balance of speed and accuracy)
def transcribe_audio(file_path):
    logging.info("Transcribing audio using Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

# Use GPT to summarize conversation
def summarize_conversation(transcript):
    prompt = f"""
    You are a medical assistant. Given the transcript of a patient-doctor conversation, summarize the interaction and list:
    1. Key health issues discussed.
    2. Medications or treatments prescribed.
    3. Any follow-up actions or next appointments.
    4. Patient concerns or questions.

    Transcript:
    {transcript}
    """

    logging.info("Summarizing conversation and extracting action items using GPT...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful and detail-oriented medical assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def save_output(text, output_file):
    with open(output_file, "w") as f:
        f.write(text)
    logging.info(f"Output saved to {output_file}")

if __name__ == "__main__":
    audio_file = "doctor_patient_audio.mp3"  # Replace with your file
    transcript = transcribe_audio(audio_file)
    
    summary_and_actions = summarize_conversation(transcript)

    save_output(summary_and_actions, "summary_and_action_items.txt")
    print("\n--- Summary and Action Items ---\n")
    print(summary_and_actions)
