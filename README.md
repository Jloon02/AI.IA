# AI.IA: Real-Time Facial Recognition and Expressive Voice Interaction

## Project Overview

**AI.IA** is an affective computing system designed to enhance human-computer interaction by detecting users' emotional expressions through webcam input and responding with emotionally-matched expressive speech.

The system integrates:
- **EmoNet** for facial emotion recognition using valence-arousal mapping.
- **EmojiVoice**, an expressive voice synthesizer trained on 13 emoji-labeled emotions.
- **Ollama**, a large language model for generating emotionally aware responses.

**Project Demo Video:** https://youtu.be/OILBhPbuMqA

---

## Project Structure
- `emojivoice/`  
  - `feel_me.py/` – TTS to expressive voice program 
  - `Matcha-TTS/` – MatchTTS models for expressive TTS - custom audio dataset: 13 emojis × 60 samples  

- `emonet/`  
  - `emonet/` – EmoNet models and training scripts
  - `trainedModel/` – Our pre-trained models, trained on AffectNet

- `src/`  
  - `predict_image.py` – Valance/Arousal prediction from given image  
  - `cnn.py` – CNN model to train on large dataset to classify valance/arousal
  - `image_dataloader.py` – Cleanse dataset to use for CNN model

- `frontend/`  
  - `App.js` – Facial emotion recognition from webcam or image  

- `emotion_mapping.csv` - Valance/Arousal to emotion mapping scheme
- `server.py` - Backend server, uses model to predict given images
- `requirements.txt` – Python dependencies 
- `README.md` – Project documentation (you’re here!)



---

## Self-Evaluation

### What We Set Out to Do
We proposed a system that:
- Detects emotions via webcam
- Maps valence/arousal to emojis and emotional states
- Synthesizes voice responses with matching emotional tone
- Generates relevant text replies

### What We Achieved
- Integrated **EmoNet** for 8-class facial emotion classification trained with AffectNet's dataset
- Created a custom dataest for **Matcha-TTS** of 780 audio samples (13 emojis × 60 samples)
- Linked **EmojiVoice** using Matcha-TTS for expressive voice synthesis
- Connected the system via a LLM **Ollama**, closing the interaction loop

### Changes From Proposal
- Shifted from training all models from scratch to **fine-tuning pre-trained models** (EmoNet, MatchTTS)
    - Note that these training and algorithms can still be found under src/model/
    - Also used AffectNet's dataset with CNN to achieve valance/arousal scores
- Recorded our own voice dataset for expressiveness instead of relying on public emotional speech datasets

### Notes for the TA
- **Voice synthesis quality** may vary due to a smaller dataset size
- **Real-time processing** was not the focus; the system runs with a modest 2-3 second delay for demonstration

---

## Dependencies and Setup

### Prerequisites
- Python 3.8+
- Webcam access

### Installation

1. Clone this repository:

```bash
git clone https://github.com/Jloon02/AI.IA.git
cd AI.IA
```

2. (Optional) Create a virtual environment:

```bash
python3 -m venv aiia_env
source aiia_env/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Ollama Install

pull the llama 3 model
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
```

5. Download dataset:

* AffectNet: https://www.kaggle.com/datasets/thienkhonghoc/affectnet

* AI.IA TTS: https://drive.google.com/drive/folders/1Esu-HPOWmTytID9lkW2cIhGXKMf9TlZp

**Note, dataset is for re-training the models. Use our pretrained models otherwise

## How to Run

Inside frontend directory to start the website, run:

```bash
npm start
```

Then to start server run:

```bash
python server.py
```

Finally to start the Emojivoice run:

```bash
python emojivoice/feel_me.py
```

This will:
* Capture a facial expression
* Classify the social signal
* Generate a relevant response from Ollama
* Speak the response using an emotionally tuned voice

## References

Mollahosseini et al. (2017). "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild"

MatchaTTS GitHub Repository: https://github.com/Matcha-TTS/Matcha-TTS

Emojivoice GitHub Repository: https://github.com/rosielab/emojivoice

Ollama: https://ollama.com