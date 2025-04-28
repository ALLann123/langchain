#!/usr/bin/python3
import os
import asyncio
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://models.inference.ai.azure.com"
)

async def entrypoint(ctx: JobContext):
    # PROPER system message format
    system_message = {
        "role": "system",
        "content": "Your name is Botnet created by Allan. Respond concisely in voice format."
    }

    await ctx.room.connect()

    
    # Initialize plugins
    vad = silero.VAD()
    stt = openai.STT()
    tts = openai.TTS()

    # Greeting
    await tts.synthesize("Ready to assist!", ctx.room)

    # Process audio
    audio_stream = ctx.room.audio_stream()
    async for audio_frame in audio_stream:
        if vad.detect(audio_frame):
            # Speech to text
            text = await stt.transcribe(audio_frame)
            print(f"User said: {text}")

            # Get LLM response (proper message format)
            response = await model.agenerate([{
                "messages": [
                    system_message,
                    {"role": "user", "content": text}
                ]
            }])
            
            # Speak response
            await tts.synthesize(response.generations[0][0].text, ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint))