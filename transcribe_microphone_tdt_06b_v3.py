import json
import tempfile
import wave
from pathlib import Path

import pyarrow  # must be imported before torch/nemo to avoid DLL conflict on Windows
# torch for model loading and inference
import torch
# numpy for audio conversion
import numpy as np
# pyaudio for microphone input
import pyaudio
# nemo_asr for Parakeet model
import nemo.collections.asr as nemo_asr
# sample rate and device settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_FRAMES = 1600  # 100ms
FRAMES_PER_BUFFER = 3200

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENERGY_THRESHOLD = 0.012
END_SILENCE_SEC = 0.7
MIN_SEGMENT_SEC = 0.8

# Hybrid recognizer class that uses energy-based endpointing and Parakeet for transcription
class TDTMicTranscriber:
    def __init__(self) -> None:
        # model and device setup (name of transcriber and CPU/CUDA)
        print(f"Loading model {MODEL_NAME} on {DEVICE}...")
        self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=MODEL_NAME, map_location=DEVICE
        )
        # eval mode for inference
        self.model.eval()
        # state for endpointing and buffering it has to catch the audio until it detects end of segment
        self._speech_started = False
        self._audio_chunks: list[np.ndarray] = []
        self._silence_samples = 0
    # feed method to accept audio bytes until endpoint
    def feed(self, data: bytes) -> str | None:
        pcm = np.frombuffer(data, dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        # rms is loudness measure for simple energy-based endpointing
        rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
        # is_speech is True when audio energy is above threshold, False when below
        # it should be replased by silero or kaldi or other more robust endpointing predictor
        is_speech = rms >= ENERGY_THRESHOLD
        # if we detect speech, we mark speech started
        if is_speech:
            self._speech_started = True
            self._silence_samples = 0
            self._audio_chunks.append(audio)
            return None
        # if we detect silence after speech we count silence samples and keep buffering audio
        if self._speech_started:
            self._audio_chunks.append(audio)
            self._silence_samples += len(audio)
            # if silence is long enough, we consider it end of segment
            if self._silence_samples >= int(SAMPLE_RATE * END_SILENCE_SEC):
                # than make segment from buffered audio
                segment = np.concatenate(self._audio_chunks) if self._audio_chunks else np.array([], dtype=np.float32)
               # after endpointing we reset state for next segment - code is below
                self._reset_segment()
                # if segment is too short, we discard it as noise/silence
                if len(segment) < int(SAMPLE_RATE * MIN_SEGMENT_SEC):
                    return None
                return self._transcribe_segment(segment)
        return None
    # flush method to be called at the end to transcribe any remaining audio in buffer
    def flush(self) -> str | None:
        # if there are no audio chunks buffered, return None
        if not self._audio_chunks:
            return None
        # if there are buffered audio chunks
        segment = np.concatenate(self._audio_chunks)
        self._reset_segment()
        # and they are less than minimum segment length
        if len(segment) < int(SAMPLE_RATE * MIN_SEGMENT_SEC):
            # return None
            return None
        # else transcribe the final segment
        return self._transcribe_segment(segment)
    # helper method to reset endpointing state for next segment
    def _reset_segment(self) -> None:
        # reset state for next segment
        self._speech_started = False
        self._audio_chunks = []
        self._silence_samples = 0

    # main transcription method that takes audio segment, saves to temp wav, runs Parakeet, and returns text
    def _transcribe_segment(self, audio: np.ndarray) -> str:
        # tempfile is used to save audio segment as wav for Parakeet input, it will be deleted after transcription
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = Path(f.name)

        try:
            # int16_audio is the audio segment converted back to int16 PCM format for wav writing
            int16_audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(int16_audio.tobytes())
            # run Parakeet with torch.no_grad() since we're only doing inference, and get the transcription result
            with torch.no_grad():
                result = self.model.transcribe([str(wav_path)], batch_size=1, verbose=False)
            # if no result, return empty string
            if not result:
                return ""
            # Parakeet's transcribe() can return a list of strings or dicts, we handle both cases to extract text
            first = result[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
            if hasattr(first, "text"):
                return str(first.text)
            return str(first)
        finally:
            try:
                # finally delete the temp wav file after transcription to clean up
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

# ── Audio stream setup ───────────────────────────────────────────────────────
def main() -> None:
    print("Parakeet TDT 0.6b-v3 minimal microphone transcriber")
    # print settings
    print(json.dumps(
        {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "chunk_frames": CHUNK_FRAMES,
            "frames_per_buffer": FRAMES_PER_BUFFER,
            "device": DEVICE,
            "model": MODEL_NAME,
            "endpoint": {
                "energy_threshold": ENERGY_THRESHOLD,
                "end_silence_sec": END_SILENCE_SEC,
                "min_segment_sec": MIN_SEGMENT_SEC,
            },
        },
        indent=2,
    ))
    # create recognizer instance
    rec = TDTMicTranscriber()
    # setup PyAudio stream for microphone input
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )
    # log then start of the stream
    print("Listening... Ctrl+C to stop")
    try:
        while True:
            # read audio data from stream which is PyAudio instance
            # I = CHUNK_FRAMES  exception_on_overflow=False means it will not raise error if audio buffer overflows
            data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            # using custom VAD method feed() processes the audio data for endpointing
            # using tempfile with NamedTemporaryFile function save procesed audio to temporary wav file
            # using TDTMicTranscriber (parakeet) instance rec transcribes the temporary wav file
            final_text = rec.feed(data)
            # if feed() returns final text
            if final_text:
                # print log with "Final: " prefix
                print(f"Final: {final_text}")
    except KeyboardInterrupt:
        print("\nStopped.")
        final_text = rec.flush()
        if final_text:
            print(f"Final: {final_text}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
