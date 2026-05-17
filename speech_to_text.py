import argparse
import sys

def transcribe_with_speechrecognition(audio_path=None, use_mic=False):
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    
    if use_mic:
        print("Listening from microphone... Please speak now.")
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.listen(source, timeout=5)
                print("Recording finished. Processing...")
        except Exception as e:
            print(f"Error accessing microphone: {e}")
            return None
    elif audio_path:
        print(f"Loading audio from {audio_path}...")
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return None
    else:
        print("Please provide an audio path or enable microphone input.")
        return None

    try:
        # Using Google Web Speech API (requires internet)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("SpeechRecognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return None


def transcribe_with_wav2vec(audio_path):
    print("Loading Hugging Face Wav2Vec2 model... (This may take a moment)")
    import torch
    import librosa
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    
    try:
        # Load pre-trained model and processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    print(f"Loading audio from {audio_path}...")
    try:
        # Wav2Vec2 typically expects 16kHz audio
        speech, rate = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None

    # Tokenize and infer
    input_values = processor(speech, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
    
    print("Transcribing...")
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    # Wav2Vec2 models often output in ALL CAPS, we can format it nicely
    return transcription.lower().capitalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Speech-to-Text System")
    parser.add_argument("--model", type=str, choices=["google", "wav2vec2"], default="google",
                        help="Choose the STT model to use: 'google' (default, uses SpeechRecognition) or 'wav2vec2' (Hugging Face local model).")
    parser.add_argument("--input", type=str, help="Path to the audio file (.wav format recommended).")
    parser.add_argument("--mic", action="store_true", help="Use microphone as input (only works with --model google).")

    args = parser.parse_args()

    if args.mic and args.model == "wav2vec2":
        print("Error: Microphone input is currently only supported with the 'google' model in this script.")
        print("For Wav2Vec2, please provide an audio file using --input.")
        sys.exit(1)

    if not args.input and not args.mic:
        print("Error: Please specify either an --input audio file or --mic for microphone input.")
        sys.exit(1)

    print(f"--- Starting Speech-to-Text ({args.model.upper()}) ---")

    result = None
    if args.model == "google":
        result = transcribe_with_speechrecognition(audio_path=args.input, use_mic=args.mic)
    elif args.model == "wav2vec2":
        result = transcribe_with_wav2vec(audio_path=args.input)

    if result:
        print("\n--- Transcription Result ---")
        print(result)
        print("----------------------------\n")
    else:
        print("\nFailed to transcribe audio.")
