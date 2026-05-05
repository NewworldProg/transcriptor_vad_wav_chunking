# Dedicated 0.6b-v3 environment setup (GTX 1070 compatible attempt)

$Python = "e:\Repoi\LocalTranscriptModular\parakeet_tdt_06b\.venv\Scripts\python.exe"

& $Python -m pip install --upgrade pip
& $Python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
& $Python -m pip install -r "e:\Repoi\LocalTranscriptModular\parakeet_tdt_06b\requirements.txt"
