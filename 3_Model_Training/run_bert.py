import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Force execution on local NVIDIA RTX 3060 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using execution device: {device}")
if torch.cuda.is_available():
    print(f"GPU Model Verified: {torch.cuda.get_device_name(0)}")

# 2. Load and isolate text arrays
file_path = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Transport dataset\Transport_Labeled.csv"
df = pd.read_csv(file_path).dropna(subset=['Cleaned_Text'])

# Map text string labels to discrete integers for PyTorch tensor compatibility
label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df['Label_ID'] = df['Sentiment_Label'].map(label_map)

X = df['Cleaned_Text'].values
y = df['Label_ID'].values

# 3. Stratified Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Tokenization Layer using BERT Base Architecture
print("Initializing BERT Tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def encode_text(text_array):
    input_ids = []
    attention_masks = []
    for text in text_array:
        encoded = tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length=128,          
            padding='max_length',    # <--- THE NEW COMMAND
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

train_ids, train_masks = encode_text(X_train)
test_ids, test_masks = encode_text(X_test)
train_labels = torch.tensor(y_train)
test_labels = torch.tensor(y_test)

# 5. Formulate DataLoaders (Using Batch Size of 8 to protect 6GB VRAM)
batch_size = 8
train_data = TensorDataset(train_ids, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_ids, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# 6. Initialize BERT Sequence Classification Architecture
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

# 7. Optimizer Settings
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# 8. Training Optimization Loop (3 Epochs)
epochs = 3
print("\nStarting Deep Learning Fine-Tuning Execution...")
for epoch in range(epochs):
    print(f"--- Epoch {epoch + 1} / {epochs} ---")
    model.train()
    total_train_loss = 0
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        model.zero_grad()        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Guard rails against gradient explosion
        optimizer.step()
    print(f"Average Training Loss Matrix: {total_train_loss / len(train_dataloader):.4f}")

# 9. Model Evaluation Phase
print("\nExecuting Blind Inference Phase on Test Matrix...")
model.eval()
predictions, true_labels = [], []

for batch in test_dataloader:
    b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    logits = outputs.logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    predictions.append(logits)
    true_labels.append(label_ids)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1)
flat_true_labels = np.concatenate(true_labels, axis=0)

# 10. Generate Output Metrics
target_names = ["Negative", "Neutral", "Positive"]
print("\n==================================================")
print("         BERT DEEP LEARNING CLASSIFICATION REPORT  ")
print("==================================================")
print(classification_report(flat_true_labels, flat_predictions, target_names=target_names, digits=4))
print("==================================================")

# 11. Plot and Save Confusion Matrix
output_dir = r"C:\Users\eucod\OneDrive\Desktop\Data Science and Analytics\Fourth Year Project\Transport dataset\Model_Results"
cm = confusion_matrix(flat_true_labels, flat_predictions)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Advanced BERT Transformer Confusion Matrix', fontweight='bold', pad=15)
plt.ylabel('Actual Ground-Truth Sentiment')
plt.xlabel('Predicted Model Sentiment')
plt.tight_layout()

cm_filename = os.path.join(output_dir, 'BERT_Confusion_Matrix.png')
plt.savefig(cm_filename, dpi=300)
plt.close()
print(f"\n✅ Deep Learning Phase Securely Complete. Matrix saved to:\n{cm_filename}")