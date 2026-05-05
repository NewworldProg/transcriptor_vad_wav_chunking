import sys
sys.path.insert(0, "..")

import tempfile
import wave
import json
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
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


@app.websocket("/ws/parakeet-stt")
async def ws_parakeet_stt(websocket: WebSocket):
    """WebSocket endpoint for streaming Parakeet transcription.
    
    Expected message format:
    - Binary: raw PCM16 mono audio chunks
    - Text: JSON commands
        {"type": "flush"} - transcribe accumulated buffer
        {"type": "reset"} - clear buffer without transcribing
    """
    await websocket.accept()
    
    if transcriber is None:
        await websocket.send_json({"error": "Transcriber not loaded"})
        await websocket.close()
        return
    
    try:
        buffer = np.array([], dtype=np.float32)
        chunk_index = 0
        
        await websocket.send_json({
            "type": "ready",
            "message": "Ready for Parakeet transcription",
            "sample_rate": 16000
        })
        
        while True:
            message = await websocket.receive()
            
            # Text message: control commands
            if "text" in message and message["text"] is not None:
                text_payload = message["text"].strip()
                
                if text_payload.lower() == "flush":
                    if buffer.size > 0:
                        try:
                            text = transcriber._transcribe_segment(buffer)
                            await websocket.send_json({
                                "type": "transcription",
                                "index": chunk_index,
                                "text": text
                            })
                            chunk_index += 1
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "detail": str(e)
                            })
                        finally:
                            buffer = np.array([], dtype=np.float32)
                    
                    await websocket.send_json({"type": "flushed"})
                
                elif text_payload.lower() == "reset":
                    buffer = np.array([], dtype=np.float32)
                    await websocket.send_json({"type": "reset_ack"})
            
            # Binary message: audio data
            elif "bytes" in message and message["bytes"] is not None:
                audio_chunk = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                if audio_chunk.size > 0:
                    buffer = np.concatenate([buffer, audio_chunk], axis=0)
    
    except WebSocketDisconnect:
        print("Client disconnected from /ws/parakeet-stt")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

