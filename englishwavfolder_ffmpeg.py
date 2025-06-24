import os
import mimetypes
import pandas as pd
import speech_recognition as sr
import pydub
from pydub import AudioSegment
import pydub.utils
import soundfile
import whisper
import json # Added import for json to catch JSONDecodeError specifically


# --- BEGIN: Add these lines and set your FFmpeg/FFprobe paths ---
# IMPORTANT: The paths below have been updated with the path you provided.
# Ensure ffprobe.exe is in the same directory as ffmpeg.exe
pydub.utils.get_prober_name = lambda: "C:\\ffmpeg\\bin\\ffmpeg.exe"
pydub.utils.get_prober_name_base = lambda: os.path.basename("C:\\ffmpeg\\bin\\ffmpeg.exe")

# Set the path for ffprobe.exe explicitly for pydub's probing functionality
pydub.utils.get_prober_name = lambda: "C:\\ffmpeg\\bin\\ffprobe.exe"
pydub.utils.get_prober_name_base = lambda: os.path.basename("C:\\ffmpeg\\bin\\ffprobe.exe")

# --- END: Add these lines ---


def check_file_types(folder_path):
    """
    Checks the file type of each file within the specified folder.

    Args:
        folder_path (str): The path to the folder to scan.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist or is not a directory.")
        return

    print(f"Scanning folder: {folder_path}\n")
    found_files = False

    try:
        # Iterate over all items (files and directories) in the specified folder
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)

            # Check if the current item is a file
            if os.path.isfile(item_path):
                found_files = True
                # Get the MIME type of the file
                mime_type, _ = mimetypes.guess_type(item_path)

                if mime_type:
                    print(f"File: {item_name} -> Type: {mime_type}")
                else:
                    # If mimetypes.guess_type doesn't find a MIME type,
                    # try to infer from the file extension
                    _, file_extension = os.path.splitext(item_name)
                    if file_extension:
                        print(f"File: {item_name} -> Type: Unknown (Extension: {file_extension})")
                    else:
                        print(f"File: {item_name} -> Type: Unknown (No extension)")
            elif os.path.isdir(item_path):
                print(f"Directory: {item_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

    if not found_files:
        print("No files found in this folder.")

def recognize_speech_from_file_offline(audio_file_path):
    """
    Recognizes speech from an audio file using an offline recognizer (Whisper).
    Converts audio to a compatible format if necessary.
    """
    if not os.path.exists(audio_file_path):
        return "Error: Audio file not found."

    # Create a temporary file path for the converted audio
    converted_audio_path = audio_file_path + "_converted.wav"

    try:
        # Load the audio file using pydub and convert to a compatible format
        try:
            audio = AudioSegment.from_file(audio_file_path)
            # Ensure the audio object is not empty/corrupted before proceeding
            if len(audio) == 0:
                return f"Error: Audio file '{os.path.basename(audio_file_path)}' resulted in empty audio segment after loading with pydub. It might be corrupted or truly empty."
        except pydub.exceptions.CouldntDecodeError:
            return f"Error: Could not decode audio file '{os.path.basename(audio_file_path)}'. Ensure FFmpeg/libav is installed and the file is not corrupted."
        except json.decoder.JSONDecodeError as json_e:
            # Catch JSONDecodeError specifically during pydub loading phase
            return f"JSON decoding error during pydub loading of '{os.path.basename(audio_file_path)}': {json_e}. This suggests an issue with ffprobe's output or corrupted file metadata."
        except Exception as e:
            # General error during pydub loading
            return f"An unexpected error occurred during pydub loading for '{os.path.basename(audio_file_path)}': {e}"


        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        try:
            audio.export(converted_audio_path, format="wav")
        except Exception as e:
            return f"An error occurred during pydub export to temporary WAV for '{os.path.basename(audio_file_path)}': {e}"


        r = sr.Recognizer()
        with sr.AudioFile(converted_audio_path) as source:
            audio_data = r.record(source)

        text = "" # Initialize text to empty string
        try:
            # Attempt to recognize speech using Whisper
            text = r.recognize_whisper(audio_data)

            # Optional: Add a warning if recognized text is empty but audio data existed
            if not text and len(audio_data.frame_data) > 0:
                print(f"Warning: recognize_whisper returned an empty string for {os.path.basename(audio_file_path)}. This might indicate silent audio or an issue with the Whisper model itself.")

        except json.decoder.JSONDecodeError as json_e:
            # This specific error usually means an unexpected, non-JSON response was received
            # during an internal call related to Whisper recognition.
            return f"JSON decoding error during Whisper recognition for '{os.path.basename(audio_file_path)}': {json_e}. This often points to an issue with the underlying Whisper model's output or a misconfigured API call if an online model was inadvertently used."
        except sr.UnknownValueError:
            return f"Could not understand audio in '{os.path.basename(audio_file_path)}'. (No speech detected or unclear audio)."
        except sr.RequestError as e:
            return f"Could not request results from recognition service for '{os.path.basename(audio_file_path)}'; {e}"
        except Exception as e:
            # Catch any other unexpected errors specifically during the recognition step
            return f"An unexpected error occurred during Whisper recognition for '{os.path.basename(audio_file_path)}': {e}"

        return text # Return the recognized text

    except Exception as e:
        # Catch any errors that fall through the specific blocks above
        return f"A top-level unexpected error occurred during processing for '{os.path.basename(audio_file_path)}': {e}"
    finally:
        # Clean up the temporary converted file
        if os.path.exists(converted_audio_path):
            os.remove(converted_audio_path)

def process_wav_files_in_folder(folder_path):
    """
    Processes all WAV files in a given folder and returns a DataFrame
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return pd.DataFrame()

    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]

    if not wav_files:
        print(f"No WAV files found in folder '{folder_path}'.")
        return pd.DataFrame()

    data = []
    for wav_file in wav_files:
        file_path = os.path.join(folder_path, wav_file)
        result = recognize_speech_from_file_offline(file_path)
        data.append({"Folder": os.path.basename(folder_path), "WAV File": wav_file, "Text": result})

    df = pd.DataFrame(data)
    return df

def process_wav_files_in_subfolders_indexed(root_folder):
    """
    Iterates through subfolders and processes WAV files, setting the folder name as index
    """
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
            df['Folder'] = os.path.basename(subfolder_path)
            df.set_index('Folder', inplace=True)
            all_data[os.path.basename(subfolder_path)] = df

    if all_data:
        combined_df = pd.concat(all_data)
        return combined_df
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    folder_path = input("Enter the folder path: ").strip()
    output_csv_name = f"all_transcriptions_indexed_{os.path.basename(folder_path) or 'root'}.csv"

    if not folder_path:
        print("Folder path is required.")
        exit(1)

    all_transcriptions_df = process_wav_files_in_subfolders_indexed(folder_path)

    if not all_transcriptions_df.empty:
        print("\nCombined Transcriptions (Indexed by Folder):")
        print(all_transcriptions_df)
        all_transcriptions_df.to_csv(output_csv_name, index=True)
        print(f"\nTranscriptions saved to: {output_csv_name}")
    else:
        print("No WAV files found in the specified folder or any of its subfolders.")
