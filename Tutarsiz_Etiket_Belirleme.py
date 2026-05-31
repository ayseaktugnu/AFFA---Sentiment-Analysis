import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

#-------Veri setini oku
df = pd.read_csv(r"C:\Users\aaktug\Desktop\DuyguAnalizi\1_orjinalVeri.csv")

#---------- CardiffNLP 3 sınıflı sentiment modeli ve tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Etiket eşlemesi 
label_map = {0: 0, 1: 1, 2: 2}  # NEGATIVE = 0, NEUTRAL = 1, POSITIVE = 2

batch_size = 32
comments = df["Cleaned_Comment"].tolist()
predicted_labels = []
confidences = []

for i in tqdm(range(0, len(comments), batch_size), desc="Tahmin Ediliyor"):
    batch_texts = comments[i:i+batch_size]
    encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1)
        batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
        batch_scores = torch.max(probs, dim=1).values.cpu().numpy()

    # Etiketlerin haritalanması
    mapped_preds = [label_map[p] for p in batch_preds]
    predicted_labels.extend(mapped_preds)
    confidences.extend(batch_scores)

#Tahminlerin dataframe'e eklenmesi

df["Predicted_Label"] = predicted_labels
df["Confidence"] = confidences

#Tahminle uyuşan satırları tut
df_cleaned = df[df["Sentiment"] == df["Predicted_Label"]]

#Temizlenmiş veriyi kaydet
df_clean.to_csv(r"C:\Users\aaktug\Desktop\DuyguAnalizi\temizlenmis_etiketli_Veri.csv", index=False)

print("Temizlenmiş veri başarıyla kaydedildi: temizlenmis_etiketli_veri.csv")