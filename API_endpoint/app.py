import sys
sys.path.insert(0, "..")

import tempfile
import wave
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

# Import existing transcriber
from transcribe_microphone_tdt_06b_v3 import TDTMicTranscriber

app = FastAPI(title="Transcriptor API", version="1.0.0")

# Global transcriber instance (loaded once at startup)
transcriber = None


@app.on_event("startup")
async def startup_event():
    global transcriber
    print("Initializing transcriber...")
    transcriber = TDTMicTranscriber()
    print("Transcriber ready")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe uploaded audio file."""
    if transcriber is None:
        raise HTTPException(status_code=503, detail="Transcriber not loaded")
    
    try:
        content = await file.read()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(content)
        
        try:
            # Load audio using built-in wave module
            with wave.open(str(tmp_path), "rb") as wf:
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe using existing transcriber
            text = transcriber._transcribe_segment(audio)
            
            return {"text": text}
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "transcriber_loaded": transcriber is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

