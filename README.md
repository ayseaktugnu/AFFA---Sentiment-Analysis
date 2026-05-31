AFFA-Integrated Hybrid Deep Learning Framework for Explainable Sentiment Analysis

Bu depo, aşağıdaki çalışmada kullanılan kaynak kodları içermektedir:

"An AFFA-Integrated Hybrid Deep Learning Framework for Explainable Sentiment Analysis"

Depo İçeriği
Veri_on_isleme.py

Veri seti ön işleme adımlarını gerçekleştirir:

Geçersiz duygu etiketlerinin kaldırılması
Boş yorumların kaldırılması
Bozuk/anlamsız karakter içeren metinlerin kaldırılması
Tüm metinlerin küçük harfe dönüştürülmesi
URL, e-posta adresi, kullanıcı adı (@username) ve emojilerin kaldırılması
Özel karakterlerin ve sayıların kaldırılması
Fazla boşlukların temizlenmesi
Üç karakterden kısa yorumların kaldırılması
Tutarsiz_Etiket_Belirleme.py

Önceden eğitilmiş CardiffNLP Twitter-RoBERTa duygu analizi modeli kullanılarak etiket tutarlılığı filtrelemesi gerçekleştirir.

Yalnızca veri setindeki orijinal etiket ile model tahmininin eşleştiği örnekler korunur.

LSTM3_Attention_AFFA.py

Önerilen AFFA-entegreli LSTM (3 Katmanlı)–Attention mimarisini içerir:

Adaptive Feature Fusion Architecture (AFFA)
Model eğitimi
Performans değerlendirmesi
Karışıklık Matrisi (Confusion Matrix)
ROC–AUC analizi
Precision–Recall AUC analizi
Gereksinimler

Python 3.10 veya üzeri

Başlıca kütüphaneler:

TensorFlow
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
PyTorch
Transformers
tqdm
Veri Seti

Bu çalışmada kullanılan veri seti, makalede belirtilen kamuya açık duygu analizi kaynaklarından elde edilmiştir.
