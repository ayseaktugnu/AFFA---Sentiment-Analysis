#-------------Vektör Çıkarma
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm
DTYPE = np.float32        
embedding_dim = 300
max_len = 30 #max 30 kelime #--30x300
df = pd.read_csv(r"C:\Users\aaktug\Desktop\eskiler\DuyguAnalizi\temizlenmis_etiketli_Veri.csv")
texts  = df["Cleaned_Comment"].astype(str).tolist()
labels = df["Sentiment"].astype(int).tolist()

#------FastText vektörlerini yükle
embedding_path = r"C:\Users\aaktug\Desktop\eskiler\DuyguAnalizi\cc.en.300.vec"
ft_model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
print("Sequence-based FastText vektör işlemi----BAŞLADI!!!!!") 

# --------- Sequence-based vektörleme (float32)
def vectorize_sequence(text, model, max_len=max_len, dim=embedding_dim, dtype=DTYPE):
    tokens = text.lower().split() # Her kelime bir token olarak yapılandırıldı
    arr = np.zeros((max_len, dim), dtype=dtype)  # PAD = 0.0 (float32)
    for i, tok in enumerate(tokens[:max_len]): #en fazla 30 token olabilir
        if tok in model:
            # gensim vektörü zaten float32 olur; garantiye alıyoruz
            arr[i] = np.asarray(model[tok], dtype=dtype)
    return arr
N = len(texts)
X = np.empty((N, max_len, embedding_dim), dtype=DTYPE)
for i, t in enumerate(tqdm(texts)):
    X[i] = vectorize_sequence(t, ft_model)
y = np.asarray(labels, dtype=np.int64)

# -------- Kaydet
np.save(r"C:\Users\aaktug\Desktop\eskiler\DuyguAnalizi\eski_X_cnn_fasttext.npy", X)
np.save(r"C:\Users\aaktug\Desktop\eskiler\DuyguAnalizi\eski_y_cnn.npy", y)

print("Sequence-based FastText vektörleme tamamlandı. X shape:", X.shape, "| dtype:", X.dtype)
print(np.isnan(X).sum())  # NaN var mı?
print(np.std(X, axis=1).mean())  #------ Vektörlerin varyansı sıfır mı? Eğer varyans sıfıra yakınsa veya her satır aynıysa, embedding bozulmuş demektir.
print("Sıfır olmayan oran:", np.count_nonzero(X) / X.size)
print("Ortalama:", np.mean(X))
print("Standart sapma:", np.std(X)) # 0.1 0.4 arası olmalı çok küçükse vektör bozulmuş
