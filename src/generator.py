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
        "Puck": "male",
        "Charon": "male",
        "Orus": "male",
        "Achird": "male",
        "Enceladus": "male",
        "Zephyr": "female",
        "Leda": "female",
        "Kore": "female",
        "Aoede": "female",
        "Gacrux": "female",
        "Sulafat": "female",
    }

    STYLES = [
        "slow",
        "cry",
        "anxious",
        "kind",
        "laugh",
        "bright",
        "commanding",
        "mellow",
        "animated"
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
        "A direct and {style} explanation presented by a {gender} speaker.",
        "An engaging {style} presentation given by a {gender} instructor.",
        "A {gender} narrator provides a {style} breakdown of the subject matter.",
        "The {gender} voice offers a {style} and academic delivery.",
        "A precise {style} lecture spoken by a {gender} academic.",
        "In a {style} manner, the {gender} speaker guides the listener through the topic.",
        "A highly articulate {gender} voice performing a {style} narration.",
        "The {style} quality of this {gender} speaker is perfect for educational content.",
        "A {gender} speaker uses an authoritative yet {style} tone.",
        "This {style} tutorial is narrated by a steady {gender} voice.",
        "A warm {gender} speaker provides a {style} instructional overview.",
        "The audio showcases a {gender} voice with a distinct {style} cadence.",
        "An articulate {style} explanation by a {gender} voice actor.",
        "A {gender} speaker adopts a {style} persona for this educational clip.",
        "This {style} delivery is performed by a clear-spoken {gender} individual.",
        "A {style} and methodical explanation from a {gender} speaker.",
        "The {gender} educator uses a {style} rhythm throughout the recording.",
        "A well-paced {style} narration delivered by a {gender} voice.",
        "A {gender} voice guides the lesson with a {style} and clear approach.",
        "The recording captures a {gender} speaker in a {style} teaching moment.",
        "A {style} and expressive {gender} voice recounts the educational material.",
        "This {gender} speaker provides a consistent {style} flow for learning.",
        "A balanced {style} tone is used by the {gender} narrator here.",
        "An insightful {style} explanation spoken by a {gender} specialist.",
        "The {gender} speaker maintains a {style} presence throughout the audio.",
        "A clear-cut {style} teaching style from a {gender} professional.",
        "This {gender} voice sounds both helpful and {style} in its delivery.",
        "A {style} pedagogical narration by a {gender} speaker.",
        "The {gender} speaker conveys complex ideas in a {style} tone.",
        "A rhythmic and {style} explanation given by a {gender} voice.",
        "This {style} auditory lesson is presented by a {gender} teacher."
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
            self.current_key_index = 0
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
        
        assigned_gender = self.VOICE_MAP[voice_name]
        
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

                # 2. Use Mapped Gender for Description
                final_description = template.format(
                    gender=assigned_gender,
                    style=style
                )
                
                # 3. Generate audio using TTS
                audio_resp = self.client.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=f"{final_description}\n\n{teacher_text}",
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

                # 4. Predicted gender for clarity check only
                predicted_gender = self._predict_gender(audio_path)
                if assigned_gender != predicted_gender:
                    logger.warning(f"Clarity: Model predicted {predicted_gender}, but using Map: {assigned_gender}")
                else:
                    logger.info(f"Clarity: Model agreed with Map ({assigned_gender})")

                metadata = {
                    "audio_file": f"voices/{sample_id}.wav",
                    "text": teacher_text,
                    "description": final_description,
                    "voice_name": voice_name,
                    "style": style,
                    "topic": topic,
                    "gender": assigned_gender
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
                logger.info(f"✅ Saved | Gender: {result['gender']} | Style: {result['style']}")
            
            # Rate limiting sleep
            time.sleep(8)
            
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
