
import pandas as pd
import re
import unicodedata

def contains_garbled_text(text):
    try:
        normalized = unicodedata.normalize('NFKD', text)
        ascii_equiv = normalized.encode('ascii', 'ignore').decode('ascii')
        return len(ascii_equiv) / len(text) < 0.5
    except:
        return True

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)


def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


#-------------Ana Temizleme Fonksiyonu ===
def clean_text(text):
    if not isinstance(text, str):
        return ''

    # Küçük harfe dönüştürme
    text = text.lower()

    # Link, e-mail, @username ve emoji temizleme
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # link
    text = re.sub(r"\S+@\S+", '', text)                  # e-mail
    text = remove_mentions(text)                         # @kullanıcı
    text = remove_emojis(text)                           # emoji

    # Özel karakter ve sayı temizleme
    text = re.sub(r"[^a-z\s]", '', text)

    # Fazla boşluk temizleme
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def preprocess_comments(df, text_column='Comment', label_column='Sentiment'):

    # 0, 1 ve 2 dışındaki duygu etiketlerini kaldır
    df = df[df[label_column].astype(str).isin(['0', '1', '2'])]

    # Gerekli sütunları al
    df = df[[text_column, label_column]].copy()

    #  Boş / eksik yorumları kaldır
    df.dropna(subset=[text_column], inplace=True)
    df = df[df[text_column].str.strip().astype(bool)]

    #  Bozuk / anlamsız karakter içeren yorumları kaldır
    df = df[~df[text_column].apply(contains_garbled_text)]

    # Metin temizleme işlemleri
    df['Cleaned_Comment'] = df[text_column].apply(clean_text)

    # Üç karakterden kısa yorumları kaldır
    df = df[df['Cleaned_Comment'].str.len() >= 3]
    return df.reset_index(drop=True)

# CSV dosyasını oku
df = pd.read_csv(r"C:\Users\aaktug\Desktop\DuyguAnalizi\1_orjinalVeri.csv")

# Ön işleme işlemini yap
df_clean = preprocess_comments(df)

# Orijinal 'Comment' sütununu sil
df_clean = df_clean.drop(columns=['Comment'])

# Temizlenmiş veriyi kaydet
df_clean.to_csv(r"C:\Users\aaktug\Desktop\DuyguAnalizi\1_ilk_temizlenen.csv", index=False)
print("Sonuç CSV dosyaya kaydedildi.")
