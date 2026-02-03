import os
import json
import time
import random
import logging
import wave
import base64
from datetime import datetime
from typing import List, Dict, Optional
from google import genai
from google.genai import types
import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSSyntheticDataGenerator:
    """
    A class to generate synthetic TTS data using Google Gemini 2.5.
    Includes automatic gender detection for generated audio.
    """
    
    VOICE_MAP = {
        "Puck": {}, "Leda": {}, "Zephyr": {}, "Kore": {}, "Charon": {},
        "Aoede": {}, "Gacrux": {}, "Achird": {}, "Sulafat": {}, "Orus": {}
    }

    STYLES = [
        "cheerful", "gentle", "energetic", "whispering", "authoritative",
        "playful", "calm", "excited", "sad", "serious", "friendly",
        "enthusiastic", "soothing", "firm", "formal", "curious",
        "happy", "angry", "fearful", "neutral"
    ]

    TEMPLATES = [
        "A {gender} speaker delivers a {style} explanation in a clear teaching voice.",
        "This recording features a {gender} voice with a {style} speaking style.",
        "A {style} narration presented by a {gender} teacher.",
        "A {gender} educator explains the topic using a {style} tone.",
        "A clear and {style} explanation spoken by a {gender} voice.",
        "This audio contains a {gender} speaker using a {style} delivery.",
        "A {style} teaching narration performed by a {gender} individual.",
        "A professional {gender} voice speaking in a {style} manner.",
        "A calm and informative {style} explanation from a {gender} speaker.",
        "A {gender} teacher presents the topic with a {style} approach.",
        "A {style} educational explanation delivered by a {gender} voice.",
        "A natural {gender} voice expressing a {style} teaching style.",
        "A {gender} narrator speaks with a {style} tone for learning purposes.",
        "A friendly {style} explanation provided by a {gender} speaker.",
        "A focused {style} teaching voice from a {gender} educator.",
        "This sample includes a {gender} voice using a {style} narration style.",
        "A structured {style} explanation spoken by a {gender} teacher.",
        "A {gender} speaker communicates the lesson in a {style} way.",
        "An educational {style} narration performed by a {gender} voice.",
        "A confident {gender} speaker delivering a {style} explanation.",
        "A simple and {style} teaching narration from a {gender} educator.",
        "A {gender} instructional voice with a {style} expression.",
        "A clear {style} lesson explained by a {gender} speaker.",
        "A composed {gender} voice presenting content in a {style} tone.",
        "A {style} learning-focused narration by a {gender} teacher.",
        "A professional educational explanation in a {style} voice by a {gender} speaker.",
        "A {gender} speaker delivers knowledge using a {style} teaching tone.",
        "A smooth and {style} explanation spoken by a {gender} voice.",
        "A {style} classroom-style narration from a {gender} educator.",
        "A direct and {style} explanation presented by a {gender} speaker."
    ]

    def __init__(self, api_keys: List[str], output_dir: str = "output"):
        if not api_keys:
            raise ValueError("At least one API key is required.")
        
        self.api_keys = api_keys
        self.current_key_index = 0
        self.output_dir = output_dir
        self.voice_dir = os.path.join(output_dir, "voices")
        self.metadata_file = os.path.join(output_dir, "metadata.jsonl")
        
        os.makedirs(self.voice_dir, exist_ok=True)
        self._initialize_client()
        self._initialize_gender_model()

    def _initialize_client(self):
        key = self.api_keys[self.current_key_index]
        self.client = genai.Client(api_key=key)
        logger.info(f"Initialized Gemini client with key {self.current_key_index + 1}/{len(self.api_keys)}")

    def _initialize_gender_model(self):
        logger.info("Loading gender classification model...")
        model_name = "prithivMLmods/Common-Voice-Geneder-Detection"
        self.gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.gender_id2label = {0: "female", 1: "male"}
        self.gender_model.eval()

    def _rotate_key(self):
        self.current_key_index += 1
        if self.current_key_index >= len(self.api_keys):
            raise RuntimeError("All API keys exhausted or quota reached.")
        self._initialize_client()

    def _predict_gender(self, audio_path: str) -> str:
        speech, _ = librosa.load(audio_path, sr=16000)
        inputs = self.gender_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.gender_model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
        return self.gender_id2label[pred]

    def _save_wav(self, filename: str, raw_data: bytes, rate: int = 24000):
        if isinstance(raw_data, str):
            raw_data = base64.b64decode(raw_data)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(raw_data)

    def generate_sample(self, topic: str, voice_name: Optional[str] = None, style: Optional[str] = None):
        """Generates a single audio sample and its metadata."""
        voice_name = voice_name or random.choice(list(self.VOICE_MAP.keys()))
        style = style or random.choice(self.STYLES)
        template = random.choice(self.TEMPLATES)
        
        sample_id = f"teacher_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        audio_path = os.path.join(self.voice_dir, f"{sample_id}.wav")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 1. Generate educational text
                text_resp = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"As a teacher feeling {style}, explain '{topic}' in 2 simple sentences for a child."
                )
                teacher_text = text_resp.text.strip()

                # 2. Generate audio using TTS
                # Use a placeholder for gender initially
                temp_desc = template.format(gender="a", style=style)
                
                audio_resp = self.client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=f"{temp_desc}\n\n{teacher_text}",
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                            )
                        )
                    )
                )

                audio_bytes = audio_resp.candidates[0].content.parts[0].inline_data.data
                self._save_wav(audio_path, audio_bytes)

                # 3. Predict gender and update description
                predicted_gender = self._predict_gender(audio_path)
                final_description = template.format(gender=predicted_gender, style=style)

                metadata = {
                    "audio_file": f"voices/{sample_id}.wav",
                    "text": teacher_text,
                    "description": final_description,
                    "voice_name": voice_name,
                    "style": style,
                    "topic": topic,
                    "gender": predicted_gender
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
                logger.info(f"✅ Saved | Gender Detected: {result['gender']}")
            
            # Rate limiting sleep
            time.sleep(2)
            
        logger.info(f"Successfully generated {samples_generated} samples.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generator.py <api_key1> <api_key2> ...")
    else:
        keys = sys.argv[1:]
        # Load topics from data/topics.json if it exists, otherwise use a default list
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "topics.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                topics = json.load(f)
        else:
            topics = ["What makes planets round?", "How do plants eat sunlight?"]
            
        gen = TTSSyntheticDataGenerator(keys)
        gen.run(topics, count=5)
