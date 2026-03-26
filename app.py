from fastapi import (
    FastAPI,
    File,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import whisper
from faster_whisper import WhisperModel as FasterWhisperModel
import io
import wave
import tempfile
import os
import numpy as np
from typing import Optional, Dict, List
import json
from pathlib import Path

app = FastAPI(title="Whisper Transcription API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WHISPER_LANGUAGES = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "aymara": "ay",
    "azerbaijani": "az",
    "bashkir": "ba",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bhojpuri": "bho",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "cebuano": "ceb",
    "chichewa": "ny",
    "chinese": "zh",
    "corsican": "co",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dhivehi": "dv",
    "dogri": "doi",
    "dutch": "nl",
    "english": "en",
    "esperanto": "eo",
    "estonian": "et",
    "ewe": "ee",
    "filipino": "tl",
    "finnish": "fi",
    "french": "fr",
    "frisian": "fy",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "haitian creole": "ht",
    "hausa": "ha",
    "hawaiian": "haw",
    "hebrew": "he",
    "hindi": "hi",
    "hmong": "hmn",
    "hungarian": "hu",
    "icelandic": "is",
    "igbo": "ig",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jv",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "kinyarwanda": "rw",
    "konkani": "gom",
    "krio": "kri",
    "kurdish": "ku",
    "kurdish (sorani)": "ckb",
    "kyrgyz": "ky",
    "lao": "lo",
    "latin": "la",
    "latvian": "lv",
    "lingala": "ln",
    "lithuanian": "lt",
    "luganda": "lg",
    "luxembourgish": "lb",
    "macedonian": "mk",
    "maithili": "mai",
    "malagasy": "mg",
    "malay": "ms",
    "malayalam": "ml",
    "maltese": "mt",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "myanmar": "my",
    "nepali": "ne",
    "norwegian": "no",
    "odia": "or",
    "pashto": "ps",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "quechua": "qu",
    "romanian": "ro",
    "russian": "ru",
    "samoan": "sm",
    "sanskrit": "sa",
    "scottish gaelic": "gd",
    "serbian": "sr",
    "sesotho": "st",
    "shona": "sn",
    "sindhi": "sd",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tajik": "tg",
    "tamil": "ta",
    "tatar": "tt",
    "telugu": "te",
    "thai": "th",
    "tigrinya": "ti",
    "tswana": "tn",
    "turkish": "tr",
    "turkmen": "tk",
    "ukrainian": "uk",
    "urdu": "ur",
    "uyghur": "ug",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "xhosa": "xh",
    "yiddish": "yi",
    "yoruba": "yo",
    "zulu": "zu",
}

MODELS = {
    "whisper": ["tiny", "base", "small", "medium", "large-v3", "large-v2"],
    "faster-whisper": ["tiny", "base", "small", "medium", "large-v3", "large-v2"],
}

model_cache = {}


def load_model(model_type: str, model_size: str, device: str = "cpu"):
    cache_key = f"{model_type}_{model_size}_{device}"
    if cache_key in model_cache:
        return model_cache[cache_key]

    if model_type == "whisper":
        model = whisper.load_model(model_size, device=device)
    elif model_type == "faster-whisper":
        compute_type = "float32" if device == "cpu" else "float16"
        model = FasterWhisperModel(model_size, device=device, compute_type=compute_type)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_cache[cache_key] = model
    return model


def transcribe_audio(
    audio_path: str, model_type: str, model_size: str, language: str = None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_type, model_size, device)

    language_code = WHISPER_LANGUAGES.get(language.lower()) if language else None

    if model_type == "whisper":
        result = model.transcribe(audio_path, language=language_code)
        return result["text"]
    else:
        segments, info = model.transcribe(audio_path, language=language_code)
        text = " ".join([segment.text for segment in segments])
        return text


@app.get("/")
async def read_root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/languages")
async def get_languages():
    return JSONResponse(content={"languages": list(WHISPER_LANGUAGES.keys())})


@app.get("/api/models")
async def get_models():
    return JSONResponse(content=MODELS)


@app.post("/api/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    model_type: str = "faster-whisper",
    model_size: str = "base",
    language: Optional[str] = None,
):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name

            audio_data = await file.read()
            temp_file.write(audio_data)
            temp_file.flush()

        transcription = transcribe_audio(temp_path, model_type, model_size, language)

        os.unlink(temp_path)

        return JSONResponse(content={"text": transcription})

    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        model_type = data.get("model_type", "faster-whisper")
        model_size = data.get("model_size", "base")
        language = data.get("language", None)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(model_type, model_size, device)
        language_code = WHISPER_LANGUAGES.get(language.lower()) if language else None

        audio_buffer = []
        sample_rate = 16000

        if model_type == "whisper":
            while True:
                data = await websocket.receive_bytes()
                audio_buffer.append(data)

                if len(audio_buffer) >= 8:
                    audio_array = np.frombuffer(
                        b"".join(audio_buffer), dtype=np.float32
                    )
                    audio_buffer.clear()

                    if len(audio_array) < 16000:
                        continue

                    audio_data = audio_array.reshape(-1, 1)

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_file:
                        temp_path = temp_file.name
                        with wave.open(temp_path, "w") as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes((audio_array * 32767).astype(np.int16))

                    result = model.transcribe(temp_path, language=language_code)

                    os.unlink(temp_path)

                    if result["text"].strip():
                        await websocket.send_json({"text": result["text"]})
        else:
            from collections import deque

            audio_queue = deque()
            CHUNK_DURATION = 3
            CHUNK_SAMPLES = sample_rate * CHUNK_DURATION

            while True:
                data = await websocket.receive_bytes()
                audio_queue.extend(np.frombuffer(data, dtype=np.float32))

                if len(audio_queue) >= CHUNK_SAMPLES:
                    chunk = np.array(list(audio_queue)[:CHUNK_SAMPLES])
                    audio_queue = deque(list(audio_queue)[CHUNK_SAMPLES:])

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".wav"
                    ) as temp_file:
                        temp_path = temp_file.name
                        with wave.open(temp_path, "w") as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes((chunk * 32767).astype(np.int16))

                    segments, info = model.transcribe(temp_path, language=language_code)
                    text = " ".join([segment.text for segment in segments])

                    os.unlink(temp_path)

                    if text.strip():
                        await websocket.send_json({"text": text})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
