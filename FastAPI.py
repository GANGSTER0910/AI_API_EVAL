import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import time, datetime
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import evaluate

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset_path = os.getenv("DATASET_PATH")

if dataset_path and os.path.isdir(dataset_path):
    print(f"Loading dataset from {dataset_path}")
    custom_dataset = load_dataset('csv', data_files={'train': os.path.join(dataset_path, 'train.csv')})
else:
    print("Loading CNN Dailymail dataset")
    custom_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

@app.get("/dataset")
async def read_dataset(file: UploadFile = File(...)):
    if not dataset_path:
        raise HTTPException(status_code=400, detail="Dataset path not configured")
    with open(f"{dataset_path}/{file.filename}", "wb") as f:
        f.write(file.file.read())
    return {"filename": file.filename}

@app.get("/test")
async def read_root():
    return {"Hello": "World"}

# Evaluation functions
def generate_text(prompt, model, tokenizer, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def evaluate_model(model_name):
    start_time = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=""  # put your huggingface token here
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Evaluation Metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    predictions, references = [], []
    for sample in custom_dataset.select(range(40)):
        print(f"Evaluating: {sample['article'][:50]}...")
        generated_text = generate_text(prompt=sample['article'], model=model, tokenizer=tokenizer)
        predictions.append(generated_text)
        references.append([sample['highlights']])

    scores = {
        "BLEU-4": bleu.compute(predictions=predictions, references=references)["bleu"],
        "ROUGE-L": rouge.compute(predictions=predictions, references=references)["rougeL"],
        "BERTScore": sum(bertscore.compute(predictions=predictions, references=references, lang="en")["f1"]) / len(predictions),
        "METEOR": meteor.compute(predictions=predictions, references=references)["meteor"],
    }

    latency = (time.time() - start_time) * 1000
    cost = 0.0004
    torch.cuda.empty_cache()
    
    return {
        "model_name": model_name,
        "scores": scores,
        "latency": latency,
        "cost": cost,
        "timestamp": datetime.now().isoformat() + "Z"
    }

@app.get("/evaluate/{model_name}")
def evaluate_api(model_name: str):
    valid_models = ['meta-llama/Llama-3.2-3B', 'tiiuae/falcon-7b']
    
    if model_name == "Llama-3.2-3B":
        model_name = valid_models[0]
    elif model_name == "falcon-7b":
        model_name = valid_models[1]
    else:
        raise HTTPException(status_code=404, detail="Model not found")
    
    result = evaluate_model(model_name)
    return [result]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
