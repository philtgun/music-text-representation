import numpy as np
import torch

from mtr.utils.audio_utils import STR_CH_FIRST, load_audio
from mtr.utils.demo_utils import get_model

framework = "contrastive"
text_type = "bert"
text_rep = "stochastic"
# load model
model, tokenizer, config = get_model(
    framework=framework,
    text_type=text_type,
    text_rep=text_rep,
)


def text_infer(query, model, tokenizer):
    text_input = tokenizer(query, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        text_embs = model.encode_bert_text(text_input, None)
    return text_embs


def audio_infer(audio_path, model, sr=16000, duration=9.91):
    audio, _ = load_audio(
        path=audio_path, ch_format=STR_CH_FIRST, sample_rate=sr, downmix_to_mono=True
    )
    input_size = int(duration * sr)
    hop = int(len(audio) // input_size) + 1
    audio = np.stack(
        [np.array(audio[i * input_size : (i + 1) * input_size]) for i in range(hop)]
    ).astype("float32")
    audio_tensor = torch.from_numpy(audio)
    with torch.no_grad():
        z_audio = model.encode_audio(audio_tensor)
    audio_embs = z_audio.mean(0).detach().cpu()
    return audio_embs


audio_path = "/path/to/audio.wav"
audio_embs = audio_infer(audio_path, model)
print(audio_embs.shape)
