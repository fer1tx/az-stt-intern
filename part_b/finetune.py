import torch, librosa, pandas as pd
from datasets import Dataset
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

MODEL_ID   = "openai/whisper-small"
DATA_DIR   = "./data/az"
OUTPUT_DIR = "./whisper-az-finetuned"

processor = WhisperProcessor.from_pretrained(MODEL_ID, language="az", task="transcribe")

df_train = pd.read_csv(f"{DATA_DIR}/train.tsv", sep="\t").dropna(subset=["sentence"])
df_dev   = pd.read_csv(f"{DATA_DIR}/dev.tsv",   sep="\t").dropna(subset=["sentence"])
df_train = df_train.sample(n=min(200, len(df_train)), random_state=42)

def load_audio(row):
    audio, _ = librosa.load(f"{DATA_DIR}/clips/{row['path']}", sr=16000)
    return {"audio": audio, "sentence": str(row["sentence"]).strip()}

def prepare(batch):
    batch["input_features"] = processor(
        batch["audio"], sampling_rate=16000,
        return_tensors="pt").input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

train_ds = Dataset.from_list([load_audio(r) for _,r in df_train.iterrows()])
dev_ds   = Dataset.from_list([load_audio(r) for _,r in df_dev.iterrows()])
train_ds = train_ds.map(prepare, remove_columns=["audio","sentence"])
dev_ds   = dev_ds.map(prepare,   remove_columns=["audio","sentence"])

@dataclass
class DataCollator:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch  = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        lbl = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lbl[:,0] == self.processor.tokenizer.bos_token_id).all():
            lbl = lbl[:,1:]
        batch["labels"] = lbl
        return batch

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32).to("cuda")
model.generation_config.language = "az"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False

metric = evaluate.load("wer")
def compute_metrics(pred):
    ids = pred.predictions
    lbl = pred.label_ids
    lbl[lbl == -100] = processor.tokenizer.pad_token_id
    return {"wer": round(metric.compute(
        predictions=processor.tokenizer.batch_decode(ids, skip_special_tokens=True),
        references=processor.tokenizer.batch_decode(lbl, skip_special_tokens=True)), 4)}

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=2,
    gradient_accumulation_steps=4, learning_rate=1e-5,
    warmup_steps=20, max_steps=200, fp16=False,
    eval_strategy="steps", eval_steps=25, save_steps=25,
    load_best_model_at_end=True, metric_for_best_model="wer",
    greater_is_better=False, predict_with_generate=True,
    generation_max_length=225, logging_steps=10,
    report_to=["none"], push_to_hub=False, dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(
    args=args, model=model,
    train_dataset=train_ds, eval_dataset=dev_ds,
    data_collator=DataCollator(processor=processor),
    compute_metrics=compute_metrics, processing_class=processor,
)
trainer.train()
print("Fine-tuning tamamdi. Checkpoint:", OUTPUT_DIR)
