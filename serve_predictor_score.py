from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# ===============================
# 1️⃣ Define the PairwiseRanker model (same as training)
# ===============================
class PairwiseRanker(nn.Module):
    def __init__(self):
        super(PairwiseRanker, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward_single(self, input_ids, attention_mask):
        """Compute score for a single text"""
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        score = self.fc(output).squeeze(-1)
        return score

    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B):
        """Keep the original dual-input structure (for backward compatibility)"""
        output_A = self.bert(input_ids=input_ids_A, attention_mask=attention_mask_A).pooler_output
        output_B = self.bert(input_ids=input_ids_B, attention_mask=attention_mask_B).pooler_output
        s_A = self.fc(output_A).squeeze(-1)
        s_B = self.fc(output_B).squeeze(-1)
        return s_A, s_B


# ===============================
# 2️⃣ Initialize FastAPI app
# ===============================
app = FastAPI(title="Prompt Score Predictor API", version="2.0")

# ===============================
# 3️⃣ Load model and tokenizer
# ===============================
MODEL_PATH = "Predictor_v3_alpaca"  # Path to model weights
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PairwiseRanker().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH} on {device}")

# ===============================
# 4️⃣ Define input schema
# ===============================
class PromptInput(BaseModel):
    prompt: str

# ===============================
# 5️⃣ Health check endpoint
# ===============================
@app.get("/")
def root():
    return {"message": "✅ Predictor score model is running!"}

# ===============================
# 6️⃣ Main endpoint: return score for a single text
# ===============================
@app.post("/score")
def get_score(input: PromptInput):
    """
    Input: a single prompt
    Output: predicted score.
    A higher score means the model predicts a longer possible response.
    """
    with torch.no_grad():
        encoding = tokenizer(input.prompt, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        score = model.forward_single(encoding["input_ids"], encoding["attention_mask"])
        score_value = float(score.item())

    return {
        "prompt": input.prompt,
        "priority": score_value
    }
