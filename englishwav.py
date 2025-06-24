import pandas as pd
import speech_recognition as sr
import pydub
import soundfile
import os
import whisper


def recognize_speech_from_file_offline(audio_file_path):
    """Recognizes speech from an audio file using an offline recognizer (Whisper)."""

    if not os.path.exists(audio_file_path):
        return "Error: Audio file not found."

    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio_data = r.record(source)

        text = r.recognize_whisper(audio_data)  # or recognize_whisper_api
        return text

    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def process_wav_files_in_folder(folder_path):
    """Processes all WAV files in a given folder and returns a DataFrame."""

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return pd.DataFrame()  # Return an empty DataFrame

    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"No WAV files found in folder '{folder_path}'.")
        return pd.DataFrame()  # Return an empty DataFrame

    data = []
    for wav_file in wav_files:
        file_path = os.path.join(folder_path, wav_file)
        result = recognize_speech_from_file_offline(file_path)
        data.append({"Folder": folder_path, "WAV File": wav_file, "Text": result})

    return pd.DataFrame(data)


# Example Usage with a test folder containing test wav files.
# create a folder named test_folder and put test.wav files inside.

 
if __name__ == "__main__":
    folder_path = input("Enter the folder path: ").strip()
    
    if not folder_path:
        print("Folder path is required.")
        exit(1)

    df = process_wav_files_in_folder(folder_path)

    if not df.empty:
        df.to_csv(f"test_transcriptions_{folder_path}.csv", index=False)
        print(df)
