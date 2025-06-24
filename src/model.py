
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import tensorflow.keras.backend as K

# --- Parameters ---
EPOCH_SAYISI = 50
class_names_all = ['NF', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
window_size = 5

# --- Utility Functions ---
def feature_selection_rf(X, y, n_features=10):
    rf = RandomForestClassifier(random_state=0)
    param_grid_rf = {'n_estimators': [100], 'max_depth': [20]}
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=2, scoring='f1_weighted', n_jobs=-1)
    grid_rf.fit(X, y)
    feature_importance = pd.Series(grid_rf.best_estimator_.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = feature_importance.head(n_features).index.tolist()
    print(f"Seçilen özellikler ({len(selected)}): {selected}")
    return selected

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, window_size, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(window_size, d_model)
    def positional_encoding(self, window_size, d_model):
        position = np.arange(window_size)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((window_size, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return tf.cast(pe, dtype=tf.float32)
    def call(self, x):
        return x + self.pos_encoding[:tf.shape(x)[1], :]

class MacroF1Callback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.X_val, self.y_val = validation_data
    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.argmax(self.model.predict(self.X_val), axis=1)
        logs['val_macro_f1'] = f1_score(self.y_val, y_pred, average='macro')

def create_time_windows(X, y, window_size=5, min_samples=5):
    X_windows, y_windows = [], []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        n_samples = len(idx)
        if n_samples < min_samples:
            for i in idx:
                window = np.repeat(X[i:i+1], window_size, axis=0)
                X_windows.append(window)
                y_windows.append(y[i])
        else:
            for i in range(len(idx) - window_size + 1):
                X_windows.append(X[idx[i:i+window_size]])
                y_windows.append(y[idx[i+window_size-1]])
    return np.array(X_windows), np.array(y_windows)

def weighted_cce(alpha, debug=False):
    def loss(y_true, y_pred):
        y_true_indices = tf.argmax(y_true, axis=-1)
        weights = tf.gather(tf.constant(alpha, dtype=tf.float32), y_true_indices)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss * weights)
    return loss

def create_transformer_model(window_size, n_features, num_classes, units=32, dropout_rate=0.3, learning_rate=0.001, num_heads=2, alpha_scaled=None):
    inputs = Input(shape=(window_size, n_features))
    x = PositionalEncoding(window_size, n_features)(inputs)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=units // num_heads)(x, x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = tf.reduce_mean(x, axis=1)
    outputs = Dense(units=num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_cce(alpha=alpha_scaled), metrics=['accuracy'])
    return model

# --- Data Loading and Preprocessing ---
def load_data():
    MAIN_PATH = 'ai4i2020.csv'  # lokal çalışma için
    df = pd.read_csv(MAIN_PATH)
    df.drop(['UDI', 'Product ID'], axis=1, inplace=True)
    return df

# --- Ana Akış ---
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(0)
    tf.random.set_seed(0)

    # 1. Veriyi yükle ve ön işle
    df = load_data()
    if df.isnull().any().any():
        raise ValueError("Veri setinde eksik değerler var!")
    conditions = [
        (df['TWF'] == 1), (df['HDF'] == 1),
        (df['PWF'] == 1), (df['OSF'] == 1), (df['RNF'] == 1)
    ]
    values = [1, 2, 3, 4, 5]
    df['Machine failure'] = np.select(conditions, values, default=0)
    df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1, inplace=True)
    df['Power'] = np.where((df['Rotational speed [rpm]'] != 0) & (df['Torque [Nm]'] != 0), df['Rotational speed [rpm]'] * df['Torque [Nm]'], 0)
    df['Power wear'] = df['Power'] * df['Tool wear [min]']
    df['Temperature difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Temperature power'] = np.where(df['Power'] != 0, df['Temperature difference'] / df['Power'], 0)
    df = pd.get_dummies(df, columns=['Type'], drop_first=True)
    X = df.drop('Machine failure', axis=1)
    y = df['Machine failure']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 2. Özellik seçimi
    selected_features = feature_selection_rf(X, y_encoded, n_features=10)
    X = X[selected_features]

    # 3. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0, stratify=y_encoded)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. ADASYN ile dengeleme
    df_train = pd.DataFrame(X_train, columns=selected_features)
    df_train['label'] = y_train
    df_maj = df_train[df_train['label'] == 0]
    df_min = df_train[df_train['label'] != 0]
    df_maj_downsampled = df_maj.sample(frac=0.5, random_state=0)
    df_balanced = pd.concat([df_maj_downsampled, df_min])
    X_train = df_balanced.drop('label', axis=1).values
    y_train = df_balanced['label'].values
    sampling_strategy = {cls: max(Counter(y_train)[cls], 1500) for cls in np.unique(y_train)}
    if 5 in np.unique(y_train):
        sampling_strategy[5] = max(sampling_strategy[5], 2000)
    adasyn = ADASYN(random_state=0, sampling_strategy=sampling_strategy, n_neighbors=3)
    X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
    X_train_res = pd.DataFrame(X_train_res, columns=selected_features).values

    # 5. Zaman pencereleme
    X_train_res, y_train_res = create_time_windows(X_train_res, y_train_res, window_size)
    X_test, y_test = create_time_windows(X_test, y_test, window_size)

    # 6. Sınıf ağırlıkları
    class_counts = Counter(y_train_res)
    total = sum(class_counts.values())
    alpha_per_class = [total / class_counts[i] for i in range(len(class_counts))]
    alpha_sum = sum(alpha_per_class)
    alpha_scaled = [a / alpha_sum for a in alpha_per_class]

    # 7. 3D reshape
    X_train_res = X_train_res.reshape(X_train_res.shape[0], window_size, len(selected_features))
    X_test = X_test.reshape(X_test.shape[0], window_size, len(selected_features))

    # 8. Eğitim/validasyon split
    X_train_res, X_val, y_train_res, y_val = train_test_split(X_train_res, y_train_res, test_size=0.2, random_state=0, stratify=y_train_res)

    # 9. Grid Search
    param_grid = {'units': [32, 64, 128], 'dropout_rate': [0.2, 0.3, 0.4], 'learning_rate': [1e-3, 5e-4], 'batch_size': [32, 64]}
    best_model, best_params, grid_search_results, history = None, None, None, None
    best_val_score = 0
    for units in param_grid['units']:
        for dropout_rate in param_grid['dropout_rate']:
            for learning_rate in param_grid['learning_rate']:
                for batch_size in param_grid['batch_size']:
                    tf.random.set_seed(0)
                    model = create_transformer_model(window_size, len(selected_features), len(np.unique(y_train_res)),
                                                    units, dropout_rate, learning_rate, 2, alpha_scaled)
                    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', restore_best_weights=True, verbose=0)
                    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
                    cb = [early_stopping, reduce_lr, MacroF1Callback(validation_data=(X_val, y_val))]
                    hist = model.fit(X_train_res, tf.keras.utils.to_categorical(y_train_res), epochs=EPOCH_SAYISI, batch_size=batch_size,
                                     validation_data=(X_val, tf.keras.utils.to_categorical(y_val)), callbacks=cb, verbose=0)
                    y_val_pred = np.argmax(model.predict(X_val), axis=1)
                    val_score = f1_score(y_val, y_val_pred, average='macro')
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_params = {'units': units, 'dropout_rate': dropout_rate, 'learning_rate': learning_rate, 'batch_size': batch_size}
                        best_model = model
                        history = hist
    print("En iyi parametreler:", best_params)

    # 10. Test Performans
    y_test_pred = np.argmax(best_model.predict(X_test), axis=1)
    f1_transformer = f1_score(y_test, y_test_pred, average='weighted')
    acc_transformer = accuracy_score(y_test, y_test_pred)
    print(f"Transformer Weighted F1: {f1_transformer:.4f}")
    print(f"Transformer Accuracy: {acc_transformer:.4f}")

    # 11. ROC Eğrisi ve SHAP (dosya: utils.py içindeki fonksiyonları çağırabilirsin)

    # Not: Plot, rapor, confusion matrix, SHAP fonksiyonlarını da utils.py'de modüler olarak yazabilirsin.

