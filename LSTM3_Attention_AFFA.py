import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

# -----------------Veriyi yükle

X = np.load(r"C:\Users\aaktug\Desktop\eskiler\DuyguAnalizi\eski_X_cnn_fasttext.npy")   # (N, 30, 300), PAD=0
y = np.load(r"C:\Users\aaktug\Desktop\eskiler\DuyguAnalizi\eski_y_cnn.npy").astype(int)
print("X shape:", X.shape)
print("y shape:", y.shape, "unique labels:", np.unique(y))
timesteps = X.shape[1]   # 30
features  = X.shape[2]   # 300
num_classes = 3

#------------------Stratified split: Train / Val / Test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)
classes = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
print("class_weight:", class_weight)


#------------------------Attention Katmanı

class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True #masking bildirilir, paddingler attentıonı etkilemez

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attn_W",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attn_b",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        e = tf.tensordot(inputs, self.W, axes=1) + self.b
        e = tf.nn.tanh(e)

        if mask is not None:
            
            mask = tf.cast(mask, tf.float32)
            e = e + (1.0 - tf.expand_dims(mask, axis=-1)) * (-1e9) 

        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(inputs * alpha, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, inputs, mask=None):
        return None

#------------Encoder-1: LSTM(3) (Attention YOK) -> V1(64)

def encoder_lstm_only(x, prefix):
    h = layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.4,
                    name=f"{prefix}_lstm1")(x)
    h = layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.4,
                    name=f"{prefix}_lstm2")(h)
    h = layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.4,
                    name=f"{prefix}_lstm3")(h)

    V = layers.Dense(64, activation="linear", name=f"{prefix}_V_64")(h)
    return V
    

#--------------------Encoder-2: LSTM(3) + Attention -> V2(64)

def encoder_lstm_attention(x, prefix):
    h = layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.4,
                    name=f"{prefix}_lstm1")(x)
    h = layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.4,
                    name=f"{prefix}_lstm2")(h)
    h = layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.4,
                    name=f"{prefix}_lstm3")(h)

    ctx = Attention(name=f"{prefix}_temporal_attention")(h)  # 64
    V = layers.Dense(64, activation="linear", name=f"{prefix}_V_64")(ctx)
    return V

# ----------------Model: AFFA

def AFFA_enc1_lstm3_enc2_lstm3att(timesteps, features, num_classes):
    inp = layers.Input(shape=(timesteps, features), name="input_fasttext")
    x = layers.Masking(mask_value=0.0, name="masking")(inp)

    V1 = encoder_lstm_only(x, prefix="enc1")
    V2 = encoder_lstm_attention(x, prefix="enc2")   

    # Skor ağı (AFFA): Dense -> ReLU -> Dense
    s1 = layers.Dense(32, activation="relu", name="score1_fc")(V1) #32 = skor ağının gizli katman boyutu
    s1 = layers.Dense(1, name="score_1")(s1)

    s2 = layers.Dense(32, activation="relu", name="score2_fc")(V2)
    s2 = layers.Dense(1, name="score_2")(s2)

    scores = layers.Concatenate(axis=1, name="scores_concat")([s1, s2])  # (B,2)
    alpha  = layers.Softmax(axis=1, name="alpha_softmax")(scores)        # (B,2)

    a1 = layers.Lambda(lambda a: tf.expand_dims(a[:, 0], axis=-1), name="alpha_1")(alpha)  # (B,1)
    a2 = layers.Lambda(lambda a: tf.expand_dims(a[:, 1], axis=-1), name="alpha_2")(alpha)  # (B,1)

    # Ağırlıklı füzyon
    V1_w = layers.Multiply(name="V1_weighted")([V1, a1])  # (B,64)
    V2_w = layers.Multiply(name="V2_weighted")([V2, a2])  # (B,64)
    F64  = layers.Add(name="fusion_F_64")([V1_w, V2_w])   # (B,64)

    # 128 projeksiyon + sınıflandırma (aynı)
    #128 olması sınıflandırıcıdan önce daha ifade gücü yüksek bir temsil alanı sağlamaktadır.
    #64 temsil öğrenme , 128 karar verme gücü
    F128 = layers.Dense(128, activation="relu", name="fusion_projection_128")(F64)
    F128 = layers.Dropout(0.3, name="dropout_after_proj")(F128)
    out  = layers.Dense(num_classes, activation="softmax", name="classifier")(F128)

    model = models.Model(inp, out, name="AFFA_Enc1_LSTM3_Enc2_LSTM3Att")
    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = AFFA_enc1_lstm3_enc2_lstm3att(timesteps, features, num_classes)
model.summary()

#---------- Eğitim-----------#

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
   
]

history = model.fit(
    X_train, y_train_cat,
    validation_split=0.1,   # train'in %10'u validation 
    epochs=150,
    batch_size=128,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

model_path = "LEARN_AFFA_LSTM3_Attention.keras"
model.save(model_path)
print(f"Model kaydedildi: {model_path}")

#---------- Değerlendirme-----------#

y_pred_probs = model.predict(X_test, batch_size=256)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"], digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg","Neu","Pos"],
            yticklabels=["Neg","Neu","Pos"])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix (AFFA: Enc1 LSTM(3), Enc2 LSTM(3)+Att)")
plt.tight_layout()
plt.show()

#---------- Macro ROC-AUC (OvR)-----------#

# y_test_cat: one-hot gerçek etiketler (N,3)
macro_auc = roc_auc_score(
    y_test_cat,
    y_pred_probs,
    multi_class='ovr',
    average='macro'
)
print(f"\n Macro ROC-AUC (OvR): {macro_auc:.4f}")

#---------- MROC Eğrileri (Her sınıf + Macro) -----------#
n_classes = num_classes
class_names = ['Negative', 'Neutral', 'Positive']
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_cat[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#-------------------Macro-average ROC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(8, 6))

# Her sınıf ROC
for i in range(n_classes):
    plt.plot(
        fpr[i], tpr[i],
        lw=2,
        label=f"{class_names[i]} (AUC={roc_auc[i]:.4f})"
    )

# Macro ROC
plt.plot(
    fpr["macro"], tpr["macro"],
    linestyle="--", lw=3,
    label=f"Macro-average (AUC={roc_auc['macro']:.4f})"
)

# Random çizgisi
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curves (OvR) - AFFA (Enc1 LSTM(3), Enc2 LSTM(3)+Att)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# --------------Precision-Recall AUC----------------#

precision = {}
recall = {}
pr_auc = {}

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(
        y_test_cat[:, i],
        y_pred_probs[:, i]
    )

    pr_auc[i] = average_precision_score(
        y_test_cat[:, i],
        y_pred_probs[:, i]
    )
    print(f"{class_names[i]} PR-AUC: {pr_auc[i]:.4f}")

    #----------Necmar testi için kaydedilenler-----------#
np.save("y_pred_affa.npy", y_pred)
np.save("y_true.npy", y_true)
print("AFFA tahminleri kaydedildi.")
