# 🎙️ TTS Synthetic Data Generation using Gemini

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange.svg)](https://ai.google.dev/)

A professional tool for generating high-quality synthetic speech datasets using Google Gemini. This project is designed to automate the creation of large-scale, structured datasets suitable for training or fine-tuning Text-to-Speech (TTS) models like Parler-TTS, VITS, and others.

## 🌟 Key Features

- 🔁 **API Key Rotation**: Automatically cycles through multiple Gemini API keys to bypass quota limits and ensure uninterrupted generation.
- 🎭 **Emotion & Style Control**: Supports a wide range of voices, styles, and emotions (e.g., cheerful, whispering, authoritative).
- 🧠 **Topic-Based Generation**: Includes a curated list of over 600 educational topics across science, space, nature, and more.
- 📊 **Structured Output**: Generates `.wav` audio files and a corresponding `metadata.jsonl` file ready for machine learning pipelines.
- 🛠️ **Modular Design**: Core logic is separated into a reusable Python class, making it easy to integrate into other projects.

## 📂 Repository Structure

```text
.
├── data/               # Static data (e.g., topics.json)
├── notebooks/          # Jupyter notebooks for interactive use
├── src/                # Core source code
│   └── generator.py    # Main TTSSyntheticDataGenerator class
├── examples/           # Sample outputs and metadata
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- One or more [Google Gemini API Keys](https://aistudio.google.com/app/apikey)

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
  "audio_file": "audio/sample_1706100000.wav",
  "text": "The sky appears blue because of Rayleigh scattering...",
  "description": "A female delivers a cheerful and medium-paced speech",
  "voice_name": "Leda",
  "style": "cheerful",
  "topic": "science",
  "gender": "female"
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed with ❤️ by [Seif Elden Osama](https://github.com/SeifEldenOsama)
