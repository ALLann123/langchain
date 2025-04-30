#!/usr/bin/python3
from elevenlabs import voices, set_api_key
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ELEVEN_LABS")

set_api_key(api_key)

# For older versions where voices() returns tuples
for voice_tuple in voices():
    # Assuming the structure is (voice_id, voice_name, ...)
    voice_name = voice_tuple[1]  # Adjust index based on actual tuple structure
    print(voice_name)