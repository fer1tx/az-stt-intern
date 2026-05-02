import torch, librosa, pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, cer
from tqdm import tqdm

MODEL_ID = "openai/whisper-small"
DATA_DIR  = "./data/az"

processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

def transcribe(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000,
                       return_tensors="pt").input_features
    inputs = inputs.to(model.device, dtype=torch.float16)
    with torch.no_grad():
        ids = model.generate(inputs, language="az",
                             task="transcribe", temperature=0.0)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

df_test = pd.read_csv(f"{DATA_DIR}/test.tsv", sep="\t").dropna(subset=["sentence"])

results = []
for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    try:
        ref  = str(row["sentence"]).strip()
        pred = transcribe(f"{DATA_DIR}/clips/{row['path']}").strip()
        results.append({"reference": ref, "predicted": pred,
                        "wer": wer(ref, pred), "cer": cer(ref, pred)})
    except Exception as e:
        print(f"Xeta: {e}")

import pandas as pd
df_r = pd.DataFrame(results)
print(f"Avg WER: {df_r['wer'].mean():.2%}")
print(f"Avg CER: {df_r['cer'].mean():.2%}")
print("\nEn yaxsi 5:")
print(df_r.nsmallest(5, 'wer')[['reference','predicted','wer','cer']])
print("\nEn pis 5:")
print(df_r.nlargest(5, 'wer')[['reference','predicted','wer','cer']])
df_r.to_csv("../results/part_a_results.csv", index=False)
