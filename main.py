import os
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydub import AudioSegment
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
    allow_origins=["*"],  # change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Helper: Deepgram STT --------
async def transcribe_audio(file_path: str) -> str:
    url = "https://api.deepgram.com/v1/listen?punctuate=true&model=nova-2"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }

    with open(file_path, "rb") as f:
        audio_data = f.read()
    
    print(f"Sending to Deepgram: {len(audio_data)} bytes, Content-Type: audio/wav")
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, content=audio_data)
        response.raise_for_status()
        data = response.json()
        transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript


# -------- Helper: Convert audio to Deepgram-compatible WAV --------
def convert_to_deepgram_wav(input_path: str, output_path: str) -> bool:
    """
    Convert audio to WAV format compatible with Deepgram:
    - 16-bit PCM (2 bytes per sample)
    - Mono channel
    - 16000 Hz sample rate (optimal for speech recognition)
    """
    try:
        # Detect actual file format by reading magic bytes
        with open(input_path, 'rb') as f:
            header = f.read(12)
            
        # Check for WebM/MKV signature
        if header[:4] == b'\x1a\x45\xdf\xa3':  # WebM signature
            print("Detected WebM format, loading with pydub...")
        
        # Load audio file - pydub auto-detects format from content
        audio = AudioSegment.from_file(input_path)
        
        print(f"Original: {audio.channels} channels, {audio.sample_width*8} bit, {audio.frame_rate} Hz")
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set sample rate to 16000 Hz (Deepgram-optimized for speech)
        audio = audio.set_frame_rate(16000)
        
        # Ensure 16-bit PCM (2 bytes per sample) - Deepgram requirement
        audio = audio.set_sample_width(2)
        
        # Export as 16-bit PCM WAV
        audio.export(
            output_path,
            format="wav",
            tags={"comment": "Deepgram STT compatible"}
        )
        
        print(f"Converted: 1 channel, 16 bit, 16000 Hz")
        return True
    except Exception as e:
        print(f"Audio conversion error: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    payload = {
        "text": text
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        audio_bytes = response.content
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_base64

# -------- Endpoint --------
@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    print(f"Received file: {audio.filename}, content_type: {audio.content_type}")
    
    # 1️⃣ Save uploaded audio temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}.{audio.filename.split('.')[-1]}"
    with open(temp_filename, "wb") as f:
        f.write(await audio.read())
    
    print(f"Saved temp file: {temp_filename}, size: {os.path.getsize(temp_filename)} bytes")
    
    # Debug: Check file format
    import subprocess
    result = subprocess.run(['file', temp_filename], capture_output=True, text=True)
    print(f"Input file format: {result.stdout}")
    
    # 2️⃣ Convert to WAV with Deepgram-compatible parameters
    wav_filename = f"{uuid.uuid4().hex}.wav"
    try:
        success = convert_to_deepgram_wav(temp_filename, wav_filename)
        if not success:
            return {"error": "Failed to convert audio to Deepgram-compatible format"}
        
        # Debug: Check converted file
        result = subprocess.run(['file', wav_filename], capture_output=True, text=True)
        print(f"Converted file format: {result.stdout}")
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

