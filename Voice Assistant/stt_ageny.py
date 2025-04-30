#!/usr/bin/env python3
import asyncio
import logging
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt
)
from livekit.plugins import assemblyai

# Load environment variables from .env file
load_dotenv()

# Configure logger for the transcriber agent
logger = logging.getLogger("transcriber")
logging.basicConfig(level=logging.INFO)

async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit agent.
    Initializes the AssemblyAI STT service and sets up event handlers.
    """
    logger.info(f"Starting transcriber (speech-to-text) for room: {ctx.room.name}")

    # Initialize the AssemblyAI STT service
    stt_impl = assemblyai.STT()

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """
        Event handler for when a new track is subscribed to in the room.
        Starts transcription if the track is an audio track.
        """
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(transcribe_track(participant, track))

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        """
        Handles the transcription of an audio track from a participant.
        Sets up the audio stream and STT stream.
        """
        audio_stream = rtc.AudioStream(track)
        stt_stream = stt_impl.stream()

        # Run audio input handling and transcription output handling concurrently
        await asyncio.gather(
            _handle_audio_input(audio_stream, stt_stream),
            _handle_transcription_output(stt_stream, participant)
        )

    async def _handle_audio_input(
        audio_stream: rtc.AudioStream, stt_stream: stt.SpeechStream
    ):
        """
        Pushes audio frames from the audio stream to the STT stream.
        """
        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

    async def _handle_transcription_output(
        stt_stream: stt.SpeechStream, participant: rtc.RemoteParticipant
    ):
        """
        Receives transcription events from the STT stream and sends them via text stream.
        """
        async for ev in stt_stream:
            if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                transcript = ev.alternatives[0].text
                print("->", transcript)
                # Send the transcript via text stream to the participant
                await participant.send_text(transcript)

    # Connect to the room with auto-subscription to audio tracks only
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

if __name__ == "__main__":
    # Run the LiveKit agent with the specified entrypoint function
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
