#!/usr/bin/python3
import assemblyai as aai
from elevenlabs import generate, stream
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

# Create the LLM
model_api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=model_api_key,
    base_url="https://models.inference.ai.azure.com"
)

class AI_Assistance:
    def __init__(self):
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.openai_client = llm
        self.elevenlabs_api_key = os.getenv("ELEVEN_LABS")  # Updated env var name

        self.transcriber = None

        # Conversation history
        self.full_transcript = [
            {"role": "system", "content": "You are a receptionist at a dental clinic. Be resourceful and efficient"},
        ]

    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            end_utterance_silence_threshold=1000
        )

        self.transcriber.connect()
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        self.transcriber.stream(microphone_stream)

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        return

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.generate_ai_response(transcript)
        else:
            print(transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        return

    def on_close(self):
        return

    def generate_ai_response(self, transcript):
        self.stop_transcription()
        self.full_transcript.append({"role": "user", "content": transcript.text})
        print(f"\nPatient: {transcript.text}\n")

        response = self.openai_client.invoke(self.full_transcript)
        ai_response = response.content

        self.generate_audio(ai_response)
        self.start_transcription()

    def generate_audio(self, text):
        self.full_transcript.append({"role": "assistant", "content": text})
        print(f"\nAI Receptionist: {text}")

        audio_stream = generate(
            api_key=self.elevenlabs_api_key,
            text=text,
            stream=True  # No voice specified; uses default
        )

        stream(audio_stream)


greeting = "Thank you for calling Nairobi dental Clinic, My name is ToothAche, how may I assist you?"

ai_assistant = AI_Assistance()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()
