from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import numpy as np
import evaluate
import nltk

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

bleu_metric = evaluate.load("bleu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=False)
model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
model.train()


save_path = "./qwen_lora_finetuned_one"
dataset = load_dataset("./datasets")



#def preprocess_function(examples):
#    inputs = ["Translate the following text: " + text for text in examples["source"]]
#    targets = examples["target"]
#    
#    model_inputs = tokenizer(inputs, max_length=20000, truncation=True, padding="max_length")
#
#    with tokenizer.as_target_tokenizer():
#        labels = tokenizer(targets, max_length=20000, truncation=True, padding="max_length")
#    model_inputs["labels"] = labels["input_ids"]
#    return model_inputs

def preprocess_long_text(examples):
    inputs = []
    labels = []
    instructions = []
    for text, translation in zip(examples["source"], examples["target"]):
        encoded_inputs = tokenizer(
            "Translate the following English text to Russian: " + text,
            max_length=512,
            truncation=True,
            stride=128,
            padding="max_length",
            return_overflowing_tokens=True
        )
        encoded_labels = tokenizer(
            translation,
            max_length=512,
            truncation=True,
            padding="max_length",
            stride=128,
            return_overflowing_tokens=True
        )
        num_fragments = min(len(encoded_inputs["input_ids"]), len(encoded_labels["input_ids"]))
        for i in range(num_fragments):
            inputs.append(encoded_inputs["input_ids"][i])
            labels.append(encoded_labels["input_ids"][i])
            instructions.append('Translate this English text to Russian')
    return {"input_ids": inputs, "labels": labels}

def compute_metrics_bleu(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #decoded_labels = [[ref] for ref in decoded_labels]
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

def compute_metrics_rouge(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    #result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #return result


full_train = True
tokenized_datasets = dataset.map(preprocess_long_text, batched=True, remove_columns=dataset["train"].column_names)
if not full_train:
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(30))
    tokenized_datasets["train"] = small_train_dataset

#tokenized_datasets["test"] = tokenized_datasets["train"].shuffle(seed=42).select(range(1))

lora_config = LoraConfig(
    r=24,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="all"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=1,
    num_train_epochs=30,
    logging_steps=10,
    save_steps=100,
    #evaluation_strategy="epoch",
    bf16=False,  
    gradient_checkpointing=True,
    #learning_rate=3e-05,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    #eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)


trainer.train()


model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)