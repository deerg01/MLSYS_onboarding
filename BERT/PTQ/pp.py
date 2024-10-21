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

df = pd.read_csv('data/IMDB_Dataset_short.csv')  


texts = df['review'].tolist()  
labels = df['sentiment'].tolist()  


labels = [1 if label == 'positive' else 0 for label in labels]


train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.25, random_state=42)

#set to use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


config = BertConfig.from_pretrained('google/bert_uncased_L-4_H-512_A-8', output_hidden_states=True, num_labels=2)
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-512_A-8')
model = BertForSequenceClassification.from_pretrained('google/bert_uncased_L-4_H-512_A-8', config=config)

model.to(device)


model.qconfig = float_qparams_weight_only_qconfig
model.train()
model_prepared = prepare_qat(model)
model_prepared.to(device)


losses = []
quantization_errors = []


optimizer = torch.optim.AdamW(model_prepared.parameters(), lr=1e-5) #Adam 사용

num_epochs = 5
for epoch in range(num_epochs):
    model_prepared.train()
    
    epoch_loss = 0.0
    correct_predictions = 0

    with tqdm(total=len(train_texts), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch') as pbar:
        for text, label in zip(train_texts, train_labels):
            optimizer.zero_grad()

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            labels_tensor = torch.tensor([label]).to(device)

            # Forward pass 
            outputs = model_prepared(**inputs, labels=labels_tensor, output_hidden_states=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (predictions == labels_tensor).sum().item()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    avg_loss = epoch_loss / len(train_texts)

    losses.append(avg_loss)

    # --------- Quantization Error ---------
    quant_error_sum = 0
    layer_count = 0
    quant_bit = '4'

    tracked_quantized_weights = []

    for name, module in model_prepared.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            original_weight = module.weight.detach().clone()

            if quant_bit == '8':
                quantized_weight = torch.quantize_per_tensor(original_weight, scale=0.1, zero_point=0, dtype=torch.qint8)
                dequantized_weight = quantized_weight.dequantize()

                min_quantized_value = quantized_weight.int_repr().min().item()
                max_quantized_value = quantized_weight.int_repr().max().item()

                print(f"Layer: {name} (8-bit Quantization)")
                print(f"Original Weight Range: {original_weight.min().item()} to {original_weight.max().item()}")
                print(f"Quantized Weight Range: {min_quantized_value} to {max_quantized_value}")
                print(f"Quantized Weight Shape: {quantized_weight.shape}")

                tracked_quantized_weights.append(quantized_weight.clone())

            elif quant_bit == '4':
                # bitsandbytes의 Linear4bit 사용하여 4비트 양자화
                if isinstance(module, torch.nn.Linear):  # Linear 레이어만 양자화
                    # 4비트 양자화된 Linear 모듈로 수정
                    quantized_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None)

                    # 원래 가중치를 복사하여 4비트 양자화된 모듈로 전달
                    quantized_module.weight.data = module.weight.data
                    quantized_weight = quantized_module.weight

                    dequantized_weight = quantized_module.weight.float()#accually not essential

                    min_quantized_value = quantized_weight.min().item()
                    max_quantized_value = quantized_weight.max().item()

                    print(f"Layer: {name}")
                    print(f"Original Weight Range: {original_weight.min().item()} to {original_weight.max().item()}")
                    print(f"Quantized Weight Range: {min_quantized_value} to {max_quantized_value}")
                    print(f"Quantized Weight Shape: {quantized_weight.shape}")

                    tracked_quantized_weights.append(quantized_weight.clone())

                else:
                    continue  # Linear 레이어가 아닌 경우 양자화하지 않음

            else:
                raise ValueError(f"Invalid quantization bit: {quant_bit}. Choose either '8' or '4'.")

            # Quantization error
            quant_error = torch.mean(torch.abs(original_weight - dequantized_weight)).item()
            quant_error_sum += quant_error
            layer_count += 1

    if layer_count > 0:
        avg_quant_error = quant_error_sum / layer_count
        quantization_errors.append(avg_quant_error)
    else:
        quantization_errors.append(0) 

# --------- Visualization ---------

# Subplots for loss, quantization error, and accuracy
fig, axs = plt.subplots(2, 1, figsize=(12, 18))

#Loss 시각화
axs[0].plot(range(1, num_epochs + 1), losses, label="Loss", marker='o', color='blue')
axs[0].set_title("Loss by Epoch")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].grid(True)
axs[0].legend()

#Quantization Error 시각화
axs[1].plot(range(1, num_epochs + 1), quantization_errors, label="Quantization Error", marker='o', color='orange')
axs[1].set_title("Quantization Error by Epoch")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Quantization Error (Mean Absolute Difference)")
axs[1].grid(True)
axs[1].legend()


plt.tight_layout()
plt.show()
