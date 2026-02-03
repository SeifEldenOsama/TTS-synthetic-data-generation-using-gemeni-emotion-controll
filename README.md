# 🎙️ TTS Synthetic Data Generation using Gemini 2.5

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.5-orange.svg)](https://ai.google.dev/)

A professional tool for generating high-quality synthetic speech datasets using **Google Gemini 2.5**. This project automates the creation of large-scale, structured datasets suitable for training or fine-tuning Text-to-Speech (TTS) models like Parler-TTS, VITS, and others.

## 🌟 Key Features

- 🚀 **Gemini 2.5 Flash & TTS**: Leverages the latest `gemini-2.5-flash` for text generation and `gemini-2.5-flash-preview-tts` for high-fidelity audio synthesis.
- 🔍 **Automatic Gender Detection**: Integrated a `Wav2Vec2` classification model to automatically detect the gender of generated audio, ensuring 100% accuracy in metadata descriptions.
- 🔁 **API Key Rotation**: Automatically cycles through multiple Gemini API keys to bypass quota limits and ensure uninterrupted generation.
- 🎭 **Emotion & Style Control**: Supports a wide range of voices and styles (e.g., cheerful, gentle, energetic, whispering, authoritative).
- 🧠 **Topic-Based Generation**: Includes a curated list of **500+ educational topics** across space, geology, history, math, and more.
- 📊 **Structured Output**: Generates `.wav` audio files and a corresponding `metadata.jsonl` file ready for machine learning pipelines.

## 📂 Repository Structure

```text
.
├── data/               # Static data (e.g., topics.json)
├── notebooks/          # Jupyter notebooks for interactive use (Colab optimized)
├── src/                # Core source code
│   └── generator.py    # Main TTSSyntheticDataGenerator class with Gender Detection
├── examples/           # Sample outputs and metadata
├── requirements.txt    # Project dependencies (including torch & transformers)
└── README.md           # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- One or more [Google Gemini API Keys](https://aistudio.google.com/app/apikey)
- FFmpeg (for audio processing via librosa)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SeifEldenOsama/TTS-synthetic-data-generation-using-gemeni-emotion-controll.git
   cd TTS-synthetic-data-generation-using-gemeni-emotion-controll
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Using the Python Script

You can run the generator directly from the command line:

```bash
python src/generator.py YOUR_API_KEY_1 YOUR_API_KEY_2
```

#### Using Jupyter Notebook

For an interactive experience, especially on Google Colab:

1. Open `notebooks/TTS_synthetic_data_generation_using_gemeni.ipynb`.
2. Follow the instructions within the notebook to mount Google Drive and provide your API keys.

## 📊 Metadata Format

The `metadata.jsonl` file follows a structured format compatible with most TTS training frameworks:

```json
{
  "audio_file": "voices/teacher_20240203_120000.wav",
  "text": "The Earth is like a giant magnet with two poles...",
  "description": "A female speaker delivers a gentle explanation in a clear teaching voice.",
  "voice_name": "Leda",
  "style": "gentle",
  "topic": "Earth & Geology",
  "gender": "female"
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed with ❤️ by [Seif Elden Osama](https://github.com/SeifEldenOsama)
