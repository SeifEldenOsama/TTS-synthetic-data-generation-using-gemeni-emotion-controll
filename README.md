# 🎙️ TTS Synthetic Data Generation using Gemini

This repository contains a **Jupyter Notebook** that generates **synthetic speech data** using **Google Gemini Text-to-Speech (TTS)**. It is designed to automatically create large-scale, structured datasets suitable for **training or fine-tuning TTS models** (e.g., Parler-TTS, VITS, etc.).

---

## 📌 What This Project Does

The notebook:

* Uses **Google Gemini TTS** to generate high-quality speech audio
* Automatically cycles through **multiple API keys** to avoid quota limits
* Generates speech for **hundreds of educational topics** (science, space, physics, etc.)
* Supports **different voices, styles, emotions, and genders**
* Saves:

  * `.wav` audio files
  * a structured `metadata.jsonl` file for ML training

This makes it ideal for:

* Synthetic dataset creation
* TTS model fine-tuning
* Voice style & emotion control experiments

---

## 📂 Output Structure

After running the notebook, the output will look like this:

```
output/
├── audio/
│   ├── sample_0001.wav
│   ├── sample_0002.wav
│   └── ...
├── metadata.jsonl
```

### `metadata.jsonl` format

Each line represents one audio sample:

```json
{
  "audio_file": "audio/sample_0001.wav",
  "text": "Why is the sky blue?",
  "description": "A female delivers a cheerful and medium-paced speech",
  "voice_name": "Leda",
  "style": "cheerful",
  "topic": "science",
  "gender": "female"
}
```

This format is **ready for direct use** in TTS training pipelines.

---

## ⚙️ Key Features in the Notebook

* 🔁 **API Key Rotation** (handles 429 / quota errors automatically)
* ⏳ **Retry & Backoff Logic** for 503 / overloaded errors
* 🎭 **Voice Style Control** (cheerful, whispering, calm, etc.)
* 🧠 **Topic-based text generation**
* 💾 **Google Drive support** (for Colab users)

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

1. Upload the notebook to Google Colab
2. Mount Google Drive when prompted
3. Add your Gemini API keys to a `.txt` file in Drive
4. Run the notebook

### Option 2: Local Jupyter

1. Install dependencies
2. Set your Gemini API key(s) as environment variables or files
3. Run the notebook cell

---
