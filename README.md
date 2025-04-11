# cvh

## AI Music Generation App

This Python-based app allows for offline AI music generation on macOS using open-source models and Hugging Face for model search and download. The app is compatible with Apple M1 hardware.

## Features

- Generate music using AI models such as OpenAI Jukebox, MuseNet, and Magenta.
- Offline functionality for music generation.
- Compatibility with macOS and Apple M1 hardware.

## Installation

1. Ensure you have Python 3.8 or later installed on your macOS system. You can download it from the official Python website.
2. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/mandaliyavatsal/cvh.git
   cd cvh
   ```
3. Create a virtual environment to manage dependencies:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
4. Install the required libraries using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the AI music generation models from the Hugging Face model hub using the `huggingface_hub` library.
2. Run the app to generate music:
   ```bash
   python app.py
   ```
3. The generated music will be saved as audio files in the output directory.

## Development Environment Setup

1. Create a `requirements.txt` file with the following content:
   ```
   torch
   transformers
   numpy
   scipy
   librosa
   soundfile
   pydub
   huggingface_hub
   ```
2. Follow the installation instructions above to set up the development environment.

## Compatibility

- The app has been tested and confirmed to work on macOS with Apple M1 hardware.
- Ensure you have the necessary permissions to read and write audio files on your system.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Resources

- [Hugging Face Model Hub](https://huggingface.co/models)
- [OpenAI Jukebox](https://github.com/openai/jukebox)
- [MuseNet](https://openai.com/blog/musenet/)
- [Magenta](https://magenta.tensorflow.org/)

