import traceback

import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())

try:
    import nemo.collections.asr as nemo_asr
    print("nemo import ok")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3",
        map_location="cuda",
    )
    print("loaded", type(model).__name__)
except Exception as exc:
    print("ERROR:", repr(exc))
    traceback.print_exc()
