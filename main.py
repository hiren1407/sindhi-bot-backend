import os
import base64
import struct
import wave
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
from openai import OpenAI
import uuid

# Load environment variables
load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_audio_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio to 16-bit mono 16kHz WAV using ffmpeg."""
    try:
        # Use ffmpeg to convert to Deepgram-compatible WAV format
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-ar', '16000',      # 16kHz sample rate
            '-ac', '1',          # Mono channel
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return False


# -------- Helper: Deepgram STT --------
async def transcribe_audio(file_path: str) -> str:
    url = "https://api.deepgram.com/v1/listen?punctuate=true&model=nova-2"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }

    with open(file_path, "rb") as f:
        audio_data = f.read()
    
    print(f"Sending to Deepgram: {len(audio_data)} bytes")
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, content=audio_data)
        response.raise_for_status()
        data = response.json()
        transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript


# -------- Helper: OpenAI Roman Sindhi --------
def convert_to_roman_sindhi(text: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = f"Translate to ROMAN SINDHI using Latin alphabet characters (A-Z). NOT Sindhi script. Output ONLY the Roman Sindhi text:\n\n{text}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    roman_sindhi = response.choices[0].message.content.strip()
    return roman_sindhi


# -------- Helper: Deepgram TTS --------
async def generate_tts(text: str) -> str:
    url = "https://api.deepgram.com/v1/speak"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        audio_bytes = response.content
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_base64


# -------- Endpoint --------
@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    print(f"Received file: {audio.filename}")
    
    # 1️⃣ Save uploaded audio temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}.webm"
    with open(temp_filename, "wb") as f:
        f.write(await audio.read())
    
    print(f"Saved temp file: {temp_filename}, size: {os.path.getsize(temp_filename)} bytes")
    
    # 2️⃣ Convert to WAV with Deepgram-compatible parameters
    wav_filename = f"{uuid.uuid4().hex}.wav"
    try:
        success = convert_audio_to_wav(temp_filename, wav_filename)
        if not success:
            return {"error": "Failed to convert audio to Deepgram-compatible format"}
        
        print(f"Converted file: {wav_filename}, size: {os.path.getsize(wav_filename)} bytes")
    except Exception as e:
        return {"error": f"Audio conversion failed: {str(e)}"}

    # 3️⃣ Deepgram STT
    try:
        transcript = await transcribe_audio(wav_filename)
    except httpx.HTTPStatusError as e:
        return {"error": f"Deepgram STT failed: {e.response.text}"}

    # 4️⃣ Convert to Roman Sindhi via OpenAI
    try:
        roman_sindhi = convert_to_roman_sindhi(transcript)
    except Exception as e:
        roman_sindhi = "[Conversion Failed]"
        print("OpenAI error:", str(e))

    # 5️⃣ Generate TTS
    try:
        audio_base64 = await generate_tts(roman_sindhi)
    except httpx.HTTPStatusError as e:
        audio_base64 = ""
        print("Deepgram TTS error:", e.response.text)

    # 6️⃣ Cleanup temp files
    try:
        os.remove(temp_filename)
        os.remove(wav_filename)
    except:
        pass

    return {
        "transcript": transcript,
        "roman_sindhi": roman_sindhi,
        "audio_base64": audio_base64
    }

