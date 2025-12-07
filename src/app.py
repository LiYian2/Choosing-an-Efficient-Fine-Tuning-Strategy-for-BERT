import gradio as gr
import time, os, torch, GPUtil
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from openai import OpenAI
from peft import PeftModel
# ================================
# ✅ DeepSeek API Config
# ================================
from config import *
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API),
    base_url=DEEPSEEK_URL
)

# ================================
# ✅ Utility: VRAM
# ================================
def get_vram():
    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        return "CPU-only"
    gpu = gpus[0]
    return f"{gpu.memoryUsed} MB / {gpu.memoryTotal} MB"

# ================================
# ✅ Global Model Registry
# ================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_INFO = {}

def load_model_info(path, name, precision="FP32", base_model_name=MODEL_NAME, num_labels=NUM_LABELS):
    """
    Auto-detect and load either a LoRA adapter or a full fine-tuned model from the given path.
    """
    # Judge if it's LoRA by checking for adapter_config.json
    is_lora = os.path.exists(os.path.join(path, "adapter_config.json"))

    if is_lora:
        print(f"  → Detected LoRA adapter at {path}")
        # 1. Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # 2. Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
    
        )
        
        # 3. Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, path)
        
    else:
        print(f"  → Loading full model from {path}")
        # 1. Load tokenizer and model directly
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=num_labels  
        )


    model = model.to(DEVICE)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / params * 100 if params > 0 else 0

    model.eval()  

    MODEL_INFO[name] = {
        "tokenizer": tokenizer,
        "model": model,
        "params": f"{params:,}",
        "trainable_percent": f"{trainable_ratio:.3f} %",
        "precision": precision
    }

# ================================
# ✅ Auto Scan Local Models
# ================================
def auto_load_models(base_dir="models"):
    for folder in os.listdir(base_dir):
        path = os.path.join(base_dir, folder)
        if os.path.isdir(path):
            try:
                load_model_info(path, folder)
                print(f"✅ Loaded: {folder}")
            except Exception as e:
                print(f"❌ Skip {folder}: {e}")

print("🚀 Auto loading local models...")
auto_load_models("../models")
print("✅ All models ready!")

# ================================
# ✅ Local Inference
# ================================
def predict_local(text, name):
    entry = MODEL_INFO[name]
    tokenizer = entry["tokenizer"]
    model = entry["model"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = softmax(logits.cpu().numpy(), axis=-1)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    return pred, conf

# ================================
# ✅ DeepSeek API Inference
# ================================
def predict_deepseek(text):
    prompt = f"""
You are a sentiment classification model.
Only respond with ONE WORD: Positive or Negative.

Text:
{text}
"""

    start = time.time()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a sentiment classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    latency = (time.time() - start) * 1000
    output = response.choices[0].message.content.strip().lower()

    pred = 1 if "positive" in output or "good" in output else 0
    conf = 1.0

    return pred, conf, latency

# ================================
# ✅ Helper
# ================================
def format_pred(pred):
    return "Positive" if pred == 1 else "Negative"

def generate_explanation(text, pred):
    return "Model detects positive sentiment." if pred == 1 else "Model detects negative sentiment."

# ================================
# ✅ Single Text Inference
# ================================
def classify(text, model_choice, history):

    if model_choice == "DeepSeek-v3 API":
        pred, conf, latency = predict_deepseek(text)
        vram = "Cloud API"
        params = "Unknown"
        trainable = "0 %"
        precision = "API"
    else:
        start = time.time()
        pred, conf = predict_local(text, model_choice)
        latency = (time.time() - start) * 1000

        vram = get_vram()
        params = MODEL_INFO[model_choice]["params"]
        trainable = MODEL_INFO[model_choice]["trainable_percent"]
        precision = MODEL_INFO[model_choice]["precision"]

    pred_label = format_pred(pred)
    explanation = generate_explanation(text, pred)

    new_row = pd.DataFrame(
        [[model_choice, text, pred_label, round(conf, 3), f"{latency:.1f} ms"]],
        columns=["Model", "Text", "Prediction", "Confidence", "Latency"]
    )

    history = pd.concat([history, new_row], ignore_index=True)

    return (
        pred_label,
        f"{conf:.3f}",
        f"{latency:.1f} ms",
        #vram,
        params,
        #trainable,
        #precision,
        explanation,
        history
    )

# ================================
# ✅ Batch Evaluation (CSV)
# ================================
def batch_evaluate(file, model_choice):
    df = pd.read_csv(file.name)
    filename = os.path.basename(file.name)

    correct, total_time = 0, 0
    total = len(df)
    results = []

    for _, row in df.iterrows():
        text = row["text"]
        label = int(row["label"])

        if model_choice == "DeepSeek-v3 API":
            pred, conf, latency = predict_deepseek(text)
        else:  # ✅ else 必须紧跟在 if 块之后
            start = time.time()
            pred, conf = predict_local(text, model_choice)
            latency = (time.time() - start) * 1000

        correct += int(pred == label)
        total_time += latency

        results.append([text, label, pred, round(conf, 3), f"{latency:.1f} ms"])

    acc = correct / total
    avg_time = total_time / total

    result_df = pd.DataFrame(
        results,
        columns=["Text", "GT Label", "Pred", "Confidence", "Latency"]
    )


    return (
        f"{acc:.4f}",
        f"{avg_time:.1f} ms",
        result_df,
        model_choice,
        filename,
        acc,
        avg_time
    )
def warmup(model, tokenizer):
    """Warm up the model with a dummy input to reduce first inference latency."""
    dummy = tokenizer("warm up", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(**dummy)

def update_batch_history(batch_history, model_choice, filename, acc, avg_time):
    new_row = pd.DataFrame([{
        "Model": model_choice,
        "File": filename,
        "Accuracy": f"{acc:.4f}",
        "Avg Inference Time": f"{avg_time:.1f} ms"
    }])
    updated = pd.concat([batch_history, new_row], ignore_index=True)
    return updated
# ================================
# ✅ Gradio UI
# ================================
if __name__ == "__main__":
    for choice in MODEL_INFO.keys():
        warmup(MODEL_INFO[choice]["model"], MODEL_INFO[choice]["tokenizer"])
    with gr.Blocks(title="Sentiment Model Evaluation Platform") as demo:
        gr.HTML("""
        <style>
        .scrollable-table {
            max-height: 200px !important;  
            overflow-y: auto !important;
        }
        .scrollable-table table {
            width: 100% !important;
            table-layout: fixed !important;
        }
        .scrollable-table::-webkit-scrollbar {
            width: 6px;
        }
        .scrollable-table::-webkit-scrollbar-thumb {
            background: #ccc;
            border-radius: 3px;
        }
        </style>
        """)

        gr.Markdown("# 🚀 PEFT vs Full Fine-tune vs API — Sentiment Evaluation Platform")

        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    choices=list(MODEL_INFO.keys()) + ["DeepSeek-v3 API"],
                    label="🧠 Choose Model",
                    value=list(MODEL_INFO.keys())[0] if MODEL_INFO else "DeepSeek-v3 API"
                )

                text_in = gr.Textbox(label="🔤 Input Text", max_lines=3)

                history = gr.Dataframe(
                    headers=["Model", "Text", "Prediction", "Confidence", "Latency"],
                    value=[],
                    label="📜 Single Inference History",
                    interactive=False,
                    wrap=True,
                    column_widths=["15%", "40%", "15%", "15%", "15%"],
                    elem_classes=["scrollable-table"],
                )

                btn = gr.Button("🔍 Classify")

                file_input = gr.File(label="📁 Upload CSV (text,label)")
                eval_btn = gr.Button("🔄 Run Batch Evaluation")

                # Batch Evaluation History Log
                gr.Markdown("### 📊 Batch Evaluation History")
                batch_history_log = gr.Dataframe(
                    value=pd.DataFrame(columns=["Model", "File", "Accuracy", "Avg Inference Time"]),
                    headers=["Model", "File", "Accuracy", "Avg Inference Time"],
                    label="📈 Batch Runs Log",
                    interactive=False,
                    wrap=True,
                    elem_classes=["scrollable-table"]
                )

            with gr.Column(scale=1):
                pred_out = gr.Textbox(label="✅ Prediction")
                conf_out = gr.Textbox(label="💯 Confidence")
                latency_out = gr.Textbox(label="⏱️ Latency")

                with gr.Accordion("⚙️ Model Information", open=False):
                    params_out = gr.Textbox(label="🧮 Total Parameters")
                    explanation_out = gr.Textbox(label="💡 Explanation")

                batch_acc = gr.Textbox(label="🎯 Batch Accuracy")
                batch_latency = gr.Textbox(label="⏱️ Avg Inference Time")
                batch_table = gr.Dataframe(
                    headers=["Text", "GT Label", "Pred", "Confidence", "Latency"],
                    label="📋 Batch Results",
                    value=[]
                )

        btn.click(
            classify,
            inputs=[text_in, model_choice, history],
            outputs=[
                pred_out,
                conf_out,
                latency_out,
                params_out,
                explanation_out,
                history,
            ]
        )

        # Modify eval_btn.click: pass in and update batch_history_log
        def run_batch_and_update_history(file, model_choice, current_batch_history):

            # Run batch evaluation and retrieve results along with metadata
            acc_str, latency_str, result_df, model, filename, acc_val, avg_time = batch_evaluate(file, model_choice)
            new_row = pd.DataFrame([{
                "Model": model,
                "File": filename,
                "Accuracy": acc_str,
                "Avg Inference Time": latency_str
            }])
            # Append the new entry to the existing history
            updated_history = pd.concat([current_batch_history, new_row], ignore_index=True)
            
            return acc_str, latency_str, result_df, updated_history

        eval_btn.click(
            run_batch_and_update_history,
            inputs=[file_input, model_choice, batch_history_log],
            outputs=[batch_acc, batch_latency, batch_table, batch_history_log]
        )

    demo.launch(share=True)