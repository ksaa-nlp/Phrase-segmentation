import sys
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, AutoModel
from fastapi import FastAPI
from pydantic import BaseModel

# Fix for torch.load inside uvicorn
sys.modules['__main__'] = sys.modules[__name__]

app = FastAPI()

# Define BiLSTM Model
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, model_name="aubmindlab/bert-base-arabertv02"):
        super(BiLSTMModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        with torch.no_grad():
            outputs = self.bert(**x)
            x = outputs.last_hidden_state[:, 1:-1, :]  # embeddings
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return self.sigmoid(output)

# Load model
device = torch.device("cpu")
lstm_model = torch.load('model_lstm.pth', map_location=device, weights_only=False)
lstm_model.to(device)

# Tokenizer
model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
tokenizer_kwargs = {
    "return_tensors": "pt",
    "padding": "max_length",
    "truncation": True,
    "max_length": 512
}

# Segmentation function
def split_sent(sent):
    raw_segments = [s.strip() for s in sent.split('.') if s.strip()]
    all_segments = []

    for raw_seg in raw_segments:
        encoded = tokenizer(raw_seg, return_offsets_mapping=True, **tokenizer_kwargs)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = lstm_model(encoded)
            prediction = (output > 0.5).float()

        indices = (prediction[0] == 1).nonzero(as_tuple=True)[0].tolist()

        if len(indices) == 0 or indices[-1] < len(offsets):
            indices.append(len(offsets))

        start_char = 0
        for idx in indices:
            end_offset = offsets[idx - 1][1] if idx > 0 else 0
            while end_offset < len(raw_seg) and raw_seg[end_offset] in ['،', ' ', '.', '؟', '؛', '!']:
                end_offset += 1
            if end_offset < start_char:
                end_offset = len(raw_seg)
            segment = raw_seg[start_char:end_offset].strip()
            if segment:
                all_segments.append(segment)
            start_char = end_offset

        if start_char < len(raw_seg):
            final_seg = raw_seg[start_char:].strip()
            if final_seg:
                all_segments.append(final_seg)

    return all_segments

# Input model (no default or example for empty field)
class InputText(BaseModel):
    text: str

# FastAPI endpoint
@app.post("/predict")
def predict(data: InputText):
    segments = split_sent(data.text)
    return {"segments": segments}
