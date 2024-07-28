import pandas as pd
import torch
import wandb
from datasets import Dataset
from peft import (LoraConfig, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments)
from trl import SFTTrainer

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
new_model = f"{base_model.split('/')[-1]}_travel_agent"


run = wandb.init(project="MTS-NLP-task-2024", job_type="training", anonymous="allow")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token


df = pd.read_csv("/home/admin/MTS-NLP-task-2024/data/train_dataset.csv")
df["output"] = df["output"] + "STOP" + "</s>"

df["text"] = df["input"] + df["output"]
df = df[
    df["text"].apply(
        lambda x: True if len(tokenizer(x)["input_ids"]) < 12000 else False
    )
]
df = df.reset_index(drop=True)
df["sample"] = 0

prev_len = len(df.loc[0, "text"])
sample = 0
for i in range(1, len(df)):
    current_len = len(df.loc[i, "text"])
    if current_len > prev_len:
        df.loc[i, "sample"] = sample
    else:
        sample += 1
        df.loc[i, "sample"] = sample
    prev_len = current_len

train_dataset = pd.DataFrame(
    {
        "text": df[df["sample"] in [1, 5, 10]]["text"],
    }
)

test_dataset = pd.DataFrame(
    {
        "text": df[df["sample"] not in [1, 5, 10]]["text"],
    }
)

train_dataset = Dataset.from_pandas(train_dataset)
test_dataset = Dataset.from_pandas(test_dataset)


model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
)
model = get_peft_model(model, peft_config)


training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_first_step=True,
    optim="paged_adamw_32bit",
    learning_rate=1e-3,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="constant",
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    max_seq_length=12000,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained(new_model)

trainer.model.push_to_hub(f"aanosov/{new_model}")
