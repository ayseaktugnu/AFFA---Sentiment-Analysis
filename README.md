AFFA-Integrated Hybrid Deep Learning Framework for Explainable Sentiment Analysis

Bu depo, aşağıdaki çalışmada kullanılan kaynak kodları içermektedir:

Depo İçeriği

Veri_on_isleme.py

Veri seti ön işleme adımlarını gerçekleştirir:

Geçersiz duygu etiketlerinin kaldırılması, boş yorumların kaldırılması, bozuk/anlamsız karakter içeren metinlerin kaldırılması, tüm metinlerin küçük harfe dönüştürülmesi, URL, e-posta adresi, kullanıcı adı (@username) ve emojilerin kaldırılması, özel karakterlerin ve sayıların kaldırılması, fazla boşlukların temizlenmesi, üç karakterden kısa yorumların kaldırılması


Tutarsiz_Etiket_Belirleme.py

Önceden eğitilmiş CardiffNLP Twitter-RoBERTa duygu analizi modeli kullanılarak etiket tutarlılığı filtrelemesi gerçekleştirilir. Yalnızca veri setindeki orijinal etiket ile model tahmininin eşleştiği örnekler korunur.

LSTM3_Attention_AFFA.py

Önerilen AFFA-entegreli LSTM (3 Katmanlı)–Attention mimarisini içerir:

Adaptive Feature Fusion Architecture (AFFA)
Model eğitimi, Performans değerlendirmesi, karışıklık matrisi (Confusion Matrix), ROC–AUC analizi, Precision–Recall AUC analizi, 

Python 3.10 veya üzeri

Başlıca kütüphaneler:

TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, PyTorch, Transformers, tqdm

Veri Seti

Bu çalışmada kullanılan veri seti, makalede belirtilen kamuya açık duygu analizi kaynaklarından elde edilmiştir.
