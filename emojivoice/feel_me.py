import os
os.environ['PHONEMIZER_ESPEAK_PATH'] = r'C:\Program Files\eSpeak NG\command_line\espeak-ng.exe'
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\command_line\libespeak-ng.dll'

from langchain_ollama import ChatOllama
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

import torch
import sounddevice as sd

from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import get_user_data_dir, intersperse, assert_model_downloaded

import emoji
import json
import numpy as np
import wavio
from pynput import keyboard
import whisper

#######################################################################################################################

VOICE = 'emoji'
TTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
############################### ASR PARAMETERS #########################################################################
SRT_PATH = "output.srt"
ASR_MODEL = "tiny.en"

############################### LLM PARAMETERS #########################################################################
LLM_MODEL = "llama3"
PROMPT = """
            You are a robot designed to help humans

            Interaction Guidelines:
            1. You are playing a build-a-story game with a human, taking turns adding one sentence each.
            2. Always respond with just one sentence.
            3. Pay special attention to emojis - they convey the user's emotional state.

            Emotions and Emojis:
            - If the user's input ends with an emoji, this represents their current emotional state
            - Your response MUST match the emotion of the user's emoji
            - If multiple emojis are present, use the last one
            - If no emoji is present, respond neutrally
            - Do not add any emojis to your response

            Allowed Emojis (we only use these):
            😎 = cool/confident
            🤔 = thoughtful/curious
            😍 = loving/admiring
            🤣 = funny/playful
            🙂 = pleasant/neutral
            😮 = surprised
            🙄 = skeptical
            😅 = nervous/awkward
            😭 = sad
            😡 = angry
            😁 = happy

            Example Interactions:
            User: "I just won the lottery! 😁"
            You: "That's amazing news! Let's buy a castle!"

            User: "My dog ran away 😭"
            You: "I'm so sorry to hear that, let's find them"

            User: "What do you think of this idea? 🤔"
            You: "It's an interesting concept worth exploring"

            User: "The meeting was rescheduled" 
            You: "We can use the extra time to prepare."
            
            Error Handling:
            - Avoid giving medical, legal, political, or financial advice. Recommend the user consult a professional instead. You can still talk about historic figures.
            
            Important Rules:
            - Never add any emoji
            - Never use symbols like ()*%&-
            - Keep responses to one sentence only
            - Match the emotion intensity of the user's emoji
        """

# Setting a higher temperature will provide more creative, but possibly less accurate answers
# Temperature ranges between 0 and 1
LLM_TEMPERATURE = 0.6

############################ TTS PARAMETERS ############################################################################
if VOICE == 'base':
    TTS_MODEL_PATH = "./Matcha-TTS/matcha_vctk.ckpt"
    SPEAKING_RATE = 0.8
    STEPS = 10
else:
    TTS_MODEL_PATH = "./Matcha-TTS/emoji-hri-paige.ckpt"
    SPEAKING_RATE = 0.8
    STEPS = 10

VOCODER_NAME = "hifigan_univ_v1"
TTS_TEMPERATURE = 0.667
VOCODER_URLS = {
    "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",
    "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",
}

emoji_mapping = {
    '😍': 107,
    '😡': 58,
    '😎': 79,
    '😭': 103,
    '🙄': 66,
    '😁': 18,
    '🙂': 12,
    '🤣': 15,
    '😮': 54,
    '😅': 22,
    '🤔': 17
}

def get_latest_emotion_emoji():
    """Read the most recent emotion and map it to an emoji."""
    file_path = '../results/emotion_results.json'

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print("Warning: JSON file is empty.")
                return None
            data = json.loads(content)

        if isinstance(data, list) and data:
            latest_entry = data[-1]
            emotion = latest_entry.get('emotion', 'neutral').lower()

            emotion_to_emoji = {
                'sad': '😭',
                'nervous': '😅',
                'skeptical': '🙄',
                'neutral': '🙂',
                'thoughtful': '🤔',
                'happy': '😁',
                'loving': '😍',
                'excited': '🤣',
                'surprise': '😮',
                'confident': '😎',
                'anger': '😡' 
                # 'fear': '😨', Not added
                # 'disgust': '🤢', Not added
            }

            return emotion_to_emoji.get(emotion, '🙂')  # Default fallback

        else:
            print("Warning: JSON file is empty or malformed.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None

def get_llm(temperature):
    """returns model instance"""
    return ChatOllama(model=LLM_MODEL, temperature=temperature)

def get_chat_prompt_template(prompt):
    """generate and return the prompt template that will answer the users query"""
    return ChatPromptTemplate(
        input_variables=["content", "messages"],
        messages=[
            SystemMessagePromptTemplate.from_template(prompt),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{content}"),
        ],
    )


def process_text(i: int, text: str, device: torch.device, play):
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["english_cleaners2"])[0], 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}

def load_matcha(checkpoint_path, device):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()
    return model

def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

def load_vocoder(vocoder_name, checkpoint_path, device):
    vocoder = None
    if vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
        vocoder = load_hifigan(checkpoint_path, device)
    else:
        raise NotImplementedError(
            f"Vocoder not implemented! define a load_<<vocoder_name>> method for it"
        )

    denoiser = Denoiser(vocoder, mode="zeros")
    return vocoder, denoiser

@torch.inference_mode()
def to_waveform(mel, vocoder, denoiser=None):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze()

    return audio.cpu().squeeze()

def play_only_synthesis(device, model, vocoder, denoiser, text, spk):
    text = text.strip()
    text_processed = process_text(0, text, device, True)

    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=STEPS,
        temperature=TTS_TEMPERATURE,
        spks=spk,
        length_scale=SPEAKING_RATE,
    )
    waveform = to_waveform(output["mel"], vocoder, denoiser)
    sd.play(waveform, 22050)
    sd.wait()

def assert_required_models_available():
    save_dir = get_user_data_dir()
    model_path = TTS_MODEL_PATH

    vocoder_path = save_dir / f"{VOCODER_NAME}"
    assert_model_downloaded(vocoder_path, VOCODER_URLS[VOCODER_NAME])
    return {"matcha": model_path, "vocoder": vocoder_path}

class Recorder:
    def __init__(self):
        self.frames = []
        self.recording = False

    def start_recording(self, filename, fs=44100, channels=1):
        self.frames = []
        self.recording = True
        stream = sd.InputStream(callback=self.callback, channels=channels, samplerate=fs)
        stream.start()
        print("Recording... Press any key but Enter to stop recording.")

        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

        stream.stop()
        stream.close()
        print("Recording stopped.")

        # Check if frames are collected
        if len(self.frames) > 0:
            # Convert frames to a NumPy array
            audio_data = np.concatenate(self.frames, axis=0)
            # Normalize audio data to fit within int16 range
            audio_data = np.clip(audio_data * 32767, -32768, 32767)
            audio_data = audio_data.astype(np.int16)  # Convert to int16

            wavio.write(filename, audio_data, fs, sampwidth=2)
        else:
            print("No audio data recorded.")

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())

    def on_press(self, key):
        self.recording = False
        return False


llm = get_llm(LLM_TEMPERATURE)
prompt = get_chat_prompt_template(PROMPT)
chain = prompt|llm

memory = ChatMessageHistory()

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="content",
    history_messages_key="messages",
)

if __name__ == "__main__":
    asr_model = whisper.load_model(ASR_MODEL)
    tts_device = torch.device(TTS_DEVICE)
    paths = assert_required_models_available()
    save_dir = get_user_data_dir()
 
    tts_model = load_matcha(paths["matcha"], tts_device)
    vocoder, denoiser = load_vocoder(VOCODER_NAME, paths["vocoder"], tts_device)

    prompt = get_chat_prompt_template(PROMPT)
    llm = get_llm(LLM_TEMPERATURE)
    chain = prompt|llm

    memory = ChatMessageHistory()
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: memory,
        input_messages_key="content",
        history_messages_key="messages",
    )

    input(f"Press Enter when you're ready to record 🎙️ ")

    recorder = Recorder()
    recorder.start_recording("output.wav")

    result = asr_model.transcribe("output.wav")
    result = result['text']

    print(f'speaker said: {result}')
    
    while True:
        if result != '':
            if "end session" in result.lower():
                exit(0)
            
            # Get current emotion emoji and append to prompt
            emotion_emoji = get_latest_emotion_emoji()
            if emotion_emoji:
                result_with_emoji = f"{result} {emotion_emoji}"
                print(f"Appended emotion emoji: {result_with_emoji}")
            else:
                result_with_emoji = result
            
            print("LLM reading")
            response = chain_with_message_history.invoke(
                {"content": result_with_emoji},
                {"configurable": {"session_id": "unused"}}
            ).content

            response = f"{response} {emotion_emoji}"
            print(response)
            
            # Get the last emoji
            emoji_list = []
            for char in response:
                if emoji.is_emoji(char):
                    emoji_list.append(char)
                    
            if VOICE == 'base':
                spk = torch.tensor([1], device=tts_device, dtype=torch.long)
            elif VOICE == 'default':
                spk = torch.tensor([7], device=tts_device, dtype=torch.long)
            else:
                spk = torch.tensor([7], device=tts_device, dtype=torch.long)
                for emote in emoji_list:
                    if emote in emoji_mapping:
                        spk = torch.tensor([emoji_mapping[emote]], device=tts_device, dtype=torch.long)
                        break
            
            response = emoji.replace_emoji(response, '')
            response = response.replace(')', '').replace('(', '')
            
            if response != '':
                play_only_synthesis(tts_device, tts_model, vocoder, denoiser, response, spk)
            else:
                play_only_synthesis(tts_device, tts_model, vocoder, denoiser, 'nice', spk)

            input(f"Press Enter when you're ready to record 🎙️ ")
            recorder = Recorder()
            recorder.start_recording("output.wav")

            result = asr_model.transcribe("output.wav")
            result = result['text']

            print(f'speaker said: {result}')
        else:
            print("I didn't hear anything, try recording again...")
            input(f"Press Enter when you're ready to record 🎙️ ")

            recorder = Recorder()
            recorder.start_recording("output.wav")

            result = asr_model.transcribe("output.wav")
            result = result['text']

            print(f'speaker said: {result}')