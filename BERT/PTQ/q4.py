import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import bitsandbytes as bnb
from torch.utils.data import DataLoader, TensorDataset

# 1. 데이터 로드
df = pd.read_csv('data/IMDB_Dataset_short.csv')
texts = df['review'].tolist()
labels = df['sentiment'].tolist()

# 레이블을 1(positive)과 0(negative)으로 변환
labels = [1 if label == 'positive' else 0 for label in labels]

# 데이터를 학습/검증 세트로 분리
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.25, random_state=42)

# 2. 토크나이저 설정 및 전처리
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess(texts):
    return tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=128)

train_encodings = preprocess(train_texts)
val_encodings = preprocess(val_texts)

# 3. 데이터 로더 설정
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 4. CUDA 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 5. 양자화 방법에 따른 모델 준비 (두 가지 양자화 방법만 사용)
methods = [
    {"quant_type": "nf4"},
    {"quant_type": "fp4"}
]

# 결과를 저장할 딕셔너리
results = {}

for method in methods:
    # 6. BERT 모델을 양자화 레이어로 감싸기
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # 교체할 레이어들을 먼저 수집
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            layers_to_replace.append((name, module))

    # 수집한 레이어들을 양자화된 레이어로 교체
    for name, module in layers_to_replace:
        quantized_linear = bnb.nn.Linear4bit(
            module.in_features, module.out_features, bias=module.bias is not None,
            quant_type=method['quant_type']
        )
        setattr(model, name, quantized_linear)

    # 모델을 CUDA 장치로 이동
    model.to(device)

    # 7. 모델 학습
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(3):  # Epoch 수는 적절히 조정 가능
        total_loss = 0
        for batch in train_loader:
            inputs, masks, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Method: {method}, Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

    # 8. 결과 저장
    results[str(method)] = total_loss / len(train_loader)

# 결과 출력
print("Quantization results: ", results)
