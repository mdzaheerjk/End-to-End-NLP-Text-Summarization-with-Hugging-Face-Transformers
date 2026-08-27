import os
import torch
from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer_path = self.config.tokenizer_path if (os.path.exists(self.config.tokenizer_path) and os.listdir(self.config.tokenizer_path)) else "google/pegasus-cnn_dailymail"
        model_path = self.config.model_path if (os.path.exists(self.config.model_path) and os.listdir(self.config.model_path)) else "google/pegasus-cnn_dailymail"
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        print(f"Loading model on {self.device} (fp16: {self.device == 'cuda'})...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print("Model loaded successfully and ready for inference!")

    def predict(self, text):
        inputs = self.tokenizer(text, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                num_beams=2,
                length_penalty=0.8,
                max_length=128,
                early_stopping=True
            )
        
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output = output.replace("<n>", " ").strip()
        
        print("Dialogue:")
        print(text)
        print("\nModel Summary:")
        print(output)
        
        return output