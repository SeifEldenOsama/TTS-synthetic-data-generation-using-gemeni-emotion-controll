import os
import json
import time
import random
import logging
from typing import List, Dict, Optional
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSSyntheticDataGenerator:
    """
    A class to generate synthetic TTS data using Google Gemini.
    Supports API key rotation and error handling.
    """
    
    VOICE_MAP = {
        "Puck": {"pitch": "medium-low"},
        "Leda": {"pitch": "medium"},
        "Zephyr": {"pitch": "low"},
        "Kore": {"pitch": "high"},
        "Charon": {"pitch": "deep"},
        "Aoede": {"pitch": "bright/melodic"},
        "Gacrux": {"pitch": "mature"},
        "Achird": {"pitch": "soft"},
        "Sulafat": {"pitch": "steady"},
        "Orus": {"pitch": "bright"}
    }

    STYLES = [
        "cheerful", "gentle", "energetic", "whispering", "slow",
        "authoritative", "playful", "calm", "excited", "sad",
        "surprised", "serious", "friendly", "cry", "enthusiastic",
        "soothing", "firm", "formal", "anxious", "curious",
        "mellow", "bright", "commanding", "kind", "animated", "laugh"
    ]

    def __init__(self, api_keys: List[str], output_dir: str = "output"):
        if not api_keys:
            raise ValueError("At least one API key is required.")
        
        self.api_keys = api_keys
        self.current_key_index = 0
        self.output_dir = output_dir
        self.voice_dir = os.path.join(output_dir, "audio")
        self.metadata_file = os.path.join(output_dir, "metadata.jsonl")
        
        os.makedirs(self.voice_dir, exist_ok=True)
        self._initialize_client()

    def _initialize_client(self):
        key = self.api_keys[self.current_key_index]
        self.client = genai.Client(api_key=key)
        logger.info(f"Initialized Gemini client with key {self.current_key_index + 1}/{len(self.api_keys)}")

    def _rotate_key(self):
        self.current_key_index += 1
        if self.current_key_index >= len(self.api_keys):
            raise RuntimeError("All API keys exhausted or quota reached.")
        self._initialize_client()

    def generate_sample(self, topic: str, voice_name: Optional[str] = None, style: Optional[str] = None):
        """Generates a single audio sample and its metadata."""
        voice_name = voice_name or random.choice(list(self.VOICE_MAP.keys()))
        style = style or random.choice(self.STYLES)
        
        description = f"A {'male' if voice_name in ['Puck', 'Zephyr', 'Charon', 'Orus'] else 'female'} delivers a {style} and medium-paced speech"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Generate text using Gemini
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=f"Generate a short educational explanation about: {topic}. Keep it under 30 words.",
                )
                generated_text = response.text.strip()
                
                
                filename = f"sample_{int(time.time() * 1000)}.wav"
                filepath = os.path.join(self.voice_dir, filename)
                
                
                metadata = {
                    "audio_file": f"audio/{filename}",
                    "text": generated_text,
                    "description": description,
                    "voice_name": voice_name,
                    "style": style,
                    "topic": topic,
                    "gender": "male" if voice_name in ['Puck', 'Zephyr', 'Charon', 'Orus'] else "female"
                }
                
                with open(self.metadata_file, "a") as f:
                    f.write(json.dumps(metadata) + "\n")
                
                return metadata
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if "429" in str(e) or "quota" in str(e).lower():
                    self._rotate_key()
                time.sleep(2 ** attempt)
        
        return None

    def run(self, topics: List[str], count: int = 10):
        """Runs the generation process for a specified number of samples."""
        samples_generated = 0
        for i in range(count):
            topic = random.choice(topics)
            logger.info(f"Generating sample {i+1}/{count} for topic: {topic}")
            result = self.generate_sample(topic)
            if result:
                samples_generated += 1
        logger.info(f"Successfully generated {samples_generated} samples.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generator.py <api_key1> <api_key2> ...")
    else:
        keys = sys.argv[1:]
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "topics.json")
        with open(data_path, "r") as f:
            topics = json.load(f)
        gen = TTSSyntheticDataGenerator(keys)
        gen.run(topics, count=5)
