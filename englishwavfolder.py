
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
        data.append({"Folder": os.path.basename(folder_path), "WAV File": wav_file, "Text": result})

    df = pd.DataFrame(data)
    return df

def process_wav_files_in_subfolders_indexed(root_folder):
    """Iterates through subfolders and processes WAV files, setting the folder name as index."""

    all_data = {}
    subfolders = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]

    if not subfolders:
        print(f"No subfolders found in '{root_folder}'.")
        return pd.DataFrame()

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)
        print(f"Processing folder: {subfolder_path}")
        df = process_wav_files_in_folder(subfolder_path)
        if not df.empty:
            df['Folder'] = os.path.basename(subfolder_path)  # Add a 'Folder' column
            df.set_index('Folder', inplace=True)  # Set 'Folder' as the index
            all_data[os.path.basename(subfolder_path)] = df

    if all_data:
        combined_df = pd.concat(all_data)
        return combined_df
    else:
        return pd.DataFrame()
    



 
if __name__ == "__main__":
    folder_path = input("Enter the folder path: ").strip()
    output_csv_name = f"all_transcriptions_indexed_{folder_path}.csv"

    all_transcriptions_df = process_wav_files_in_subfolders_indexed(folder_path)
    
    if not folder_path:
        print("Folder path is required.")
        exit(1)

    df = process_wav_files_in_folder(folder_path)

    if not all_transcriptions_df.empty:
        print("\nCombined Transcriptions (Indexed by Folder):")
        print(all_transcriptions_df)
        all_transcriptions_df.to_csv(output_csv_name, index=True)
        print(f"\nTranscriptions saved to: {output_csv_name}")
    else:
        print("No WAV files found in any subfolders.")

    