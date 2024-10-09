import math
import json
import os
import numpy as np
import gradio as gr
from tensorflow import keras
from huggingface_hub import hf_hub_download
import librosa


# Download the model
model_path = hf_hub_download(repo_id='ruben09/music_genre_classification', filename='music_genre_model.h5')

# Load the model
model = keras.models.load_model(model_path)

def process_audio(audio_file):
  map_labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
  SR = 22050
  TD = 30
  SPT = SR * TD
  num_segments = 3
  n_fft=2048
  hop_length=512

  summed_predictions = np.zeros(len(map_labels))

  sample_per_segment = int(SPT / num_segments)
  num_spectrogram_per_segment = math.ceil(sample_per_segment / hop_length)

  signal, sr = librosa.load(audio_file, sr=SR)

  for d in range(num_segments):
    start = sample_per_segment * d
    finish = start + sample_per_segment

    spectrogram = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    spectrogram_db = spectrogram.T
    if len(spectrogram_db) == num_spectrogram_per_segment:
      input_data = np.array(spectrogram_db)
      input_data = input_data[None,..., np.newaxis]
      input_data = np.transpose(input_data, (0, 2, 1, 3))
      prediction = model.predict(input_data)
      summed_predictions += prediction[0]

  averaged_predictions = summed_predictions / num_segments

  # Get the final prediction (the class with the highest probability)
  final_prediction_idx = np.argmax(averaged_predictions)
  final_class_label = map_labels[final_prediction_idx]
  final_probability = averaged_predictions[final_prediction_idx]

  # Format the result as a dictionary
  result = {
      "final_prediction": final_class_label,
      "confidence": round(float(final_probability), 2),
      "all_probabilities": {map_labels[i]: round(float(prob), 2) for i, prob in enumerate(averaged_predictions)}
  }

  return result

iface = gr.Interface(
    fn=process_audio,  # The function to process the uploaded audio
    inputs=gr.Audio(type="filepath", label="Upload Audio (WAV, MP3, FLAC)"),  # Accept audio input
    outputs="json",  # Return predictions as text
    title="Audio Classification",  # Title of the interface
    description="Upload an audio file (max 30 seconds) to get a genre classification."
)

# Launch the Gradio app
iface.launch()