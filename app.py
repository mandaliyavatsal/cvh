import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import scipy
import librosa
import soundfile as sf
from pydub import AudioSegment
from huggingface_hub import hf_hub_download
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import os

# Function to download model from Hugging Face
def download_model(model_name, output_dir):
    model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin", cache_dir=output_dir)
    return model_path

# Function to load model and tokenizer
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to generate music
def generate_music(model, tokenizer, prompt, max_length=1000):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    music = tokenizer.decode(output[0], skip_special_tokens=True)
    return music

# Function to save music as audio file
def save_music(music, filename):
    audio = AudioSegment.from_file(music, format="wav")
    audio.export(filename, format="wav")

# Function for audio analysis and manipulation
def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beat_frames

# Function to handle download button click
def handle_download(model_name, output_dir, status_label, progress_bar):
    try:
        status_label.config(text="Downloading model...")
        progress_bar.start()
        model_path = download_model(model_name, output_dir)
        status_label.config(text="Model downloaded successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        progress_bar.stop()

# Function to handle generate button click
def handle_generate(model_name, prompt, output_dir, status_label, progress_bar):
    try:
        status_label.config(text="Generating music...")
        progress_bar.start()
        model, tokenizer = load_model(model_name)
        music = generate_music(model, tokenizer, prompt)
        output_filename = os.path.join(output_dir, "generated_music.wav")
        save_music(music, output_filename)
        status_label.config(text="Music generated and saved successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        progress_bar.stop()

# Function to handle save button click
def handle_save(output_dir, status_label, progress_bar):
    try:
        status_label.config(text="Saving music...")
        progress_bar.start()
        output_filename = os.path.join(output_dir, "generated_music.wav")
        save_music("generated_music.wav", output_filename)
        status_label.config(text="Music saved successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        progress_bar.stop()

# Function to select output directory
def select_output_directory(output_dir_entry):
    output_dir = filedialog.askdirectory()
    output_dir_entry.delete(0, tk.END)
    output_dir_entry.insert(0, output_dir)

# Main function
def main():
    root = tk.Tk()
    root.title("AI Music Generation App")

    # Model name input
    tk.Label(root, text="Model Name:").grid(row=0, column=0, padx=10, pady=10)
    model_name_entry = tk.Entry(root)
    model_name_entry.grid(row=0, column=1, padx=10, pady=10)

    # Prompt input
    tk.Label(root, text="Prompt:").grid(row=1, column=0, padx=10, pady=10)
    prompt_entry = tk.Entry(root)
    prompt_entry.grid(row=1, column=1, padx=10, pady=10)

    # Output directory input
    tk.Label(root, text="Output Directory:").grid(row=2, column=0, padx=10, pady=10)
    output_dir_entry = tk.Entry(root)
    output_dir_entry.grid(row=2, column=1, padx=10, pady=10)
    tk.Button(root, text="Select", command=lambda: select_output_directory(output_dir_entry)).grid(row=2, column=2, padx=10, pady=10)

    # Status label
    status_label = tk.Label(root, text="")
    status_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

    # Progress bar
    progress_bar = ttk.Progressbar(root, mode="indeterminate")
    progress_bar.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

    # Download button
    tk.Button(root, text="Download Model", command=lambda: threading.Thread(target=handle_download, args=(model_name_entry.get(), output_dir_entry.get(), status_label, progress_bar)).start()).grid(row=5, column=0, padx=10, pady=10)

    # Generate button
    tk.Button(root, text="Generate Music", command=lambda: threading.Thread(target=handle_generate, args=(model_name_entry.get(), prompt_entry.get(), output_dir_entry.get(), status_label, progress_bar)).start()).grid(row=5, column=1, padx=10, pady=10)

    # Save button
    tk.Button(root, text="Save Music", command=lambda: threading.Thread(target=handle_save, args=(output_dir_entry.get(), status_label, progress_bar)).start()).grid(row=5, column=2, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
