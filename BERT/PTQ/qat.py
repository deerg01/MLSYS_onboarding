import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BitsAndBytesConfig
from torch.ao.quantization import float_qparams_weight_only_qconfig, prepare_qat
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm  
import numpy as np

from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.nn import Linear4bit

# 데이터셋 불러오기 (CSV 파일 경로 지정)
df = pd.read_csv('data/IMDB_Dataset.csv')  

# 데이터 전처리
texts = df['review'].tolist()  # 'review' 열에서 텍스트를 가져옴
labels = df['sentiment'].tolist()  # 'sentiment' 열에서 레이블을 가져옴

# 레이블을 0, 1로 변환 (예: 긍정 = 1, 부정 = 0)
labels = [1 if label == 'positive' else 0 for label in labels]

# 훈련 및 검증 데이터셋 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.25, random_state=42)

# 디바이스 설정 (CPU로 강제 설정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# BERT 모델 및 토크나이저 불러오기 (config를 통해 hidden states 출력 활성화)
config = BertConfig.from_pretrained('google/bert_uncased_L-4_H-512_A-8', output_hidden_states=True, num_labels=2)
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-512_A-8')
model = BertForSequenceClassification.from_pretrained('google/bert_uncased_L-4_H-512_A-8', config=config)

model.to(device)

# 양자화 준비
model.qconfig = float_qparams_weight_only_qconfig
model.train()
model_prepared = prepare_qat(model)
model_prepared.to(device)

# 시각화 부분을 위한 변수 초기화
losses = []
quantization_errors = []

# 옵티마이저 설정
optimizer = torch.optim.AdamW(model_prepared.parameters(), lr=1e-5) #Adam 사용

num_epochs = 5
for epoch in range(num_epochs):
    model_prepared.train()
    
    # 에폭별 손실 초기화
    epoch_loss = 0.0
    correct_predictions = 0

    # tqdm을 사용하여 훈련 과정 표시
    with tqdm(total=len(train_texts), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
        for text, label in zip(train_texts, train_labels):
            optimizer.zero_grad()

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            labels_tensor = torch.tensor([label]).to(device)

            # Forward pass (output_hidden_states=True를 추가)
            outputs = model_prepared(**inputs, labels=labels_tensor, output_hidden_states=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # 손실 및 정확도 업데이트
            epoch_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels_tensor).sum().item()

            # 진행 상황 업데이트
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    # 평균 손실 및 정확도 계산
    avg_loss = epoch_loss / len(train_texts)

    # 리스트에 저장
    losses.append(avg_loss)

    # ---- Quantization Error 계산 (에폭별 평균) ----
    quant_error_sum = 0
    layer_count = 0
    quant_bit = '4'

    for name, module in model_prepared.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            original_weight = module.weight.detach().clone()

            if quant_bit == '8':
                quantized_weight = torch.quantize_per_tensor(original_weight, scale=0.1, zero_point=0, dtype=torch.qint8)
                dequantized_weight = quantized_weight.dequantize()
            elif quant_bit == '4':
                if isinstance(module, torch.nn.Linear): #only processing Linear layer 
                    quantized_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None)
                
                    quantized_module.weight.data = module.weight.data
                    quantized_weight = quantized_module.weight
                    dequantized_weight = quantized_module.weight.float()
                else:
                    continue  # Linear 레이어가 아닌 경우 양자화하지 않음
                
                
            else:
                raise ValueError(f"Invalid quantization bit: {quant_bit}. Choose either '8' or '4'.")
            
            # Quantization error 계산 (원본 weight와 dequantized weight 간 차이)
            quant_error = torch.mean(torch.abs(original_weight - dequantized_weight)).item()
            quant_error_sum += quant_error
            layer_count += 1

    # 에폭별 평균 양자화 오류 계산 및 저장
    if layer_count > 0:
        avg_quant_error = quant_error_sum / layer_count
        quantization_errors.append(avg_quant_error)
    else:
        quantization_errors.append(0)  # 레이어가 없는 경우를 대비

# ---- 시각화 부분 ----

# Subplots for loss, quantization error, and accuracy
fig, axs = plt.subplots(2, 1, figsize=(12, 18))

# 첫 번째 subplot: Loss 시각화
axs[0].plot(range(1, num_epochs + 1), losses, label="Loss", marker='o', color='blue')
axs[0].set_title("Loss by Epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].grid(True)
axs[0].legend()

# 두 번째 subplot: Quantization Error 시각화
axs[1].plot(range(1, num_epochs + 1), quantization_errors, label="Quantization Error", marker='o', color='orange')
axs[1].set_title("Quantization Error by Epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Quantization Error (Mean Absolute Difference)")
axs[1].grid(True)
axs[1].legend()


# 그래프 전체 표시
plt.tight_layout()
plt.show()
