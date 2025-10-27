import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import keras_tuner as kt
import joblib
import os

# --- CONFIGURATION ---

FLOOD_DATA_TRAIN = os.path.join('..', 'data', 'flood_data_train.csv')
FLOOD_DATA_VALIDATION = os.path.join('..', 'data', 'flood_data_validation.csv')
FLOOD_DATA_TEST = os.path.join('..', 'data', 'flood_data_test.csv')

STATIC_FEATURES = ['elevation', 'slope', 'aspect', 'clay_content']
DYNAMIC_FEATURES_PREFIX = 'precip_day_'
TARGET_VARIABLE = 'flooded'
TIME_STEPS = 14 # Phải khớp với TIME_WINDOW_DAYS trong prepare_data.py

# --- 1. DATA LOADING AND PREPROCESSING ---

def load_and_preprocess_data(filepath):
    """Tải và tiền xử lý dữ liệu từ tệp CSV."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        print(f"Cảnh báo: Tệp {filepath} không tồn tại hoặc rỗng. Bỏ qua.")
        return None, None, None
        
    df = pd.read_csv(filepath)
    if df.empty:
        print(f"Cảnh báo: DataFrame từ {filepath} rỗng. Bỏ qua.")
        return None, None, None

    # LỖI ĐÃ SỬA: Phân tách chính xác X và y
    y = df[TARGET_VARIABLE]
    X_static = df[STATIC_FEATURES]
    
    # LỖI ĐÃ SỬA: Tìm tất cả các cột mưa động
    dynamic_cols = sorted([col for col in df.columns if col.startswith(DYNAMIC_FEATURES_PREFIX)])
    
    # Kiểm tra xem chúng ta có đúng số lượng cột mưa không
    if len(dynamic_cols) != TIME_STEPS:
        print(f"Lỗi: Tìm thấy {len(dynamic_cols)} cột động, nhưng mong đợi {TIME_STEPS} cho tệp {filepath}")
        return None, None, None
        
    X_dynamic = df[dynamic_cols]
    
    # Định hình lại dữ liệu động cho LSTM: [samples, timesteps, features]
    # Ở đây, 'features' là 1 (chỉ có lượng mưa)
    X_dynamic_reshaped = X_dynamic.values.reshape(-1, TIME_STEPS, 1)
    
    return X_static, X_dynamic_reshaped, y

print("Đang tải và tiền xử lý dữ liệu...")
X_static_train, X_dynamic_train, y_train = load_and_preprocess_data(FLOOD_DATA_TRAIN)
X_static_val, X_dynamic_val, y_val = load_and_preprocess_data(FLOOD_DATA_VALIDATION)
X_static_test, X_dynamic_test, y_test = load_and_preprocess_data(FLOOD_DATA_TEST)

# Kiểm tra nếu dữ liệu không tải được
# Sửa lỗi: Cần kiểm tra riêng lẻ vì val hoặc test có thể rỗng do cấu hình
if y_train is None:
    print("Lỗi: Không thể tải dữ liệu huấn luyện (flood_data_train.csv). Dừng chương trình.")
    exit()
if y_val is None:
    print("Cảnh báo: Không thể tải dữ liệu xác thực (flood_data_validation.csv).")
if y_test is None:
    print("Lỗi: Không thể tải dữ liệu kiểm tra (flood_data_test.csv). Dừng chương trình.")
    exit()

# Nếu không có dữ liệu xác thực, thì không thể chạy tuner.search
if y_val is None:
    print("Lỗi: Cần dữ liệu xác thực (flood_data_validation.csv) để chạy Keras Tuner. Dừng chương trình.")
    exit()


# Chuẩn hóa các đặc trưng tĩnh
scaler = StandardScaler()
X_static_train_scaled = scaler.fit_transform(X_static_train)
X_static_val_scaled = scaler.transform(X_static_val)
X_static_test_scaled = scaler.transform(X_static_test)

# Lưu scaler cho ứng dụng dự báo
joblib.dump(scaler, 'static_feature_scaler.pkl')
print("Scaler đã được lưu vào static_feature_scaler.pkl")

# --- 2. MODEL ARCHITECTURE AND HYPERPARAMETER TUNING (Priority 2) ---

def build_model(hp):
    """Xây dựng một mô hình LSTM có thể tinh chỉnh bằng Keras Tuner."""
    hp_lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=128, step=32)
    hp_lstm_units_2 = hp.Int('lstm_units_2', min_value=32, max_value=128, step=32)
    hp_dense_units = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    dynamic_input = Input(shape=(TIME_STEPS, 1), name='dynamic_input')
    static_input = Input(shape=(len(STATIC_FEATURES),), name='static_input')

    lstm_out = LSTM(units=hp_lstm_units_1, return_sequences=True)(dynamic_input)
    lstm_out = LSTM(units=hp_lstm_units_2)(lstm_out)

    concatenated = concatenate([lstm_out, static_input])

    x = Dense(units=hp_dense_units, activation='relu')(concatenated)
    x = Dropout(hp_dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[dynamic_input, static_input], outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_auc", direction="max"),
    max_epochs=50,
    factor=3,
    directory='keras_tuner_dir',
    project_name='flood_prediction',
    overwrite=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Giám sát val_loss
    patience=10,
    restore_best_weights=True
)

print("\n--- Bắt đầu Tìm kiếm Siêu tham số ---")
tuner.search(
    [X_dynamic_train, X_static_train_scaled], 
    y_train,
    epochs=50,
    validation_data=([X_dynamic_val, X_static_val_scaled], y_val),
    callbacks=[early_stopping]
)

best_hps_list = tuner.get_best_hyperparameters(num_trials=1)
if not best_hps_list:
    raise ValueError("Không tìm thấy siêu tham số nào. Quá trình tìm kiếm có thể đã thất bại.")
# LỖI ĐÃ SỬA: get_best_hyperparameters trả về một danh sách, lấy phần tử đầu tiên
best_hps = best_hps_list[0]

print(f"""
--- Tìm kiếm Siêu tham số Hoàn tất ---
Các Siêu tham số Tốt nhất được Tìm thấy:
- LSTM Units 1: {best_hps.get('lstm_units_1')}
- LSTM Units 2: {best_hps.get('lstm_units_2')}
- Dense Units: {best_hps.get('dense_units')}
- Dropout Rate: {best_hps.get('dropout_rate')}
- Learning Rate: {best_hps.get('learning_rate')}
""")

# --- 3. FINAL MODEL TRAINING ---

print("\n--- Huấn luyện Mô hình Cuối cùng với các Siêu tham số Tốt nhất ---")
final_model = tuner.hypermodel.build(best_hps)

# Kết hợp tập train và val để huấn luyện cuối cùng
X_dynamic_full_train = np.concatenate([X_dynamic_train, X_dynamic_val])
X_static_full_train_scaled = np.concatenate([X_static_train_scaled, X_static_val_scaled])
y_full_train = np.concatenate([y_train, y_val])

# Sử dụng validation_split trên tập dữ liệu kết hợp
final_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', # Giám sát val_loss được tạo từ validation_split
    patience=10,
    restore_best_weights=True
)

history = final_model.fit(
    [X_dynamic_full_train, X_static_full_train_scaled],
    y_full_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1, # Sử dụng 10% của tập (train+val) để dừng sớm
    callbacks=[final_early_stopping]
)

final_model.save('flood_prediction_model.keras')
print("\nMô hình cuối cùng đã được lưu vào flood_prediction_model.keras")

# --- 4. OBJECTIVE MODEL EVALUATION (Priority 3) ---

print("\n--- Đánh giá Mô hình trên Sự kiện Kiểm tra Giữ lại Chưa từng thấy ---")
y_pred_proba = final_model.predict([X_dynamic_test, X_static_test_scaled]).flatten()
y_pred_class = (y_pred_proba > 0.5).astype(int)

print("\nBáo cáo Phân loại:")
print(classification_report(y_test, y_pred_class, target_names=['Không ngập', 'Có ngập']))

print("\nMa trận Nhầm lẫn:")
print(confusion_matrix(y_test, y_pred_class))

# Tính toán ROC AUC
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nĐiểm ROC AUC: {roc_auc:.4f}")
except ValueError as e:
    print(f"\nKhông thể tính ROC AUC (có thể chỉ có một lớp trong tập test): {e}")

print("\n--- Hoàn tất ---")
