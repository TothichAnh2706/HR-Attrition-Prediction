import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

print("=== BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN & XUẤT MODEL ===")

# 1. Đọc dữ liệu
DATA_PATH = "attritioon.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(DATA_PATH)
print(f"1. Đã đọc dữ liệu từ {DATA_PATH}, kích thước: {df.shape}")

# 2. Loại bỏ các cột không mang giá trị dự đoán
cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 3. Phân loại cột
cat_cols = [c for c in df_clean.columns if df_clean[c].dtype == 'object' and c != 'Attrition']
num_cols = [c for c in df_clean.columns if df_clean[c].dtype != 'object' and c != 'Attrition']

# 4. Tìm các cột số bị lệch phải (skew > 0.75) để áp dụng log1p
skewed_cols = []
for c in num_cols:
    if df_clean[c].skew() > 0.75 and (df_clean[c] >= 0).all():
        skewed_cols.append(c)
print(f"2. Các cột áp dụng Log1p (skew > 0.75): {skewed_cols}")

# 5. Tách features X và target y
X = df_clean.drop('Attrition', axis=1)
y = df_clean['Attrition'].map({'No': 0, 'Yes': 1})

# Chia train / test (stratified 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Tính toán tỷ lệ ratio bù mất cân bằng lớp
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"3. Tỷ lệ bù mất cân bằng lớp (ratio = non-attrition / attrition): {ratio:.2f}")

# 7. Định nghĩa Tiền xử lý (ColumnTransformer)
ordinal_features = ["BusinessTravel", "OverTime"]
ordinal_categories = [
    ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    ["No", "Yes"]
]
ordinal_features2 = ["MaritalStatus"]
ordinal_categories2 = [["Single", "Married", "Divorced"]]

nominal_features = ["Department", "EducationField", "JobRole"]
binary_features = ["Gender"]

preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalEncoder(categories=ordinal_categories), ordinal_features),
        ("ordinal2", OrdinalEncoder(categories=ordinal_categories2), ordinal_features2),
        ("binary", OrdinalEncoder(), binary_features),
        ("nominal", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), nominal_features),
    ],
    remainder="passthrough",
)

# 8. Hàm biến đổi log1p trên các cột skewed
def transform_features(df_input):
    df_out = df_input.copy()
    for col in skewed_cols:
        if col in df_out.columns:
            df_out[col] = np.log1p(df_out[col])
    return df_out

X_train_transformed = transform_features(X_train)
X_test_transformed = transform_features(X_test)

# Fit preprocessor
X_train_encoded = preprocessor.fit_transform(X_train_transformed)
X_test_encoded = preprocessor.transform(X_test_transformed)

feature_names = preprocessor.get_feature_names_out()
print(f"4. Số đặc trưng sau khi mã hóa: {len(feature_names)}")

# 9. Huấn luyện mô hình tối ưu XGBoost (dựa trên kết quả thực nghiệm từ colap.ipynb)
best_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    eval_metric='logloss',
    random_state=42
)

best_model.fit(X_train_encoded, y_train)
print("5. Đã huấn luyện mô hình XGBoost thành công!")

# 10. Tìm ngưỡng tối ưu (Decision Threshold) trên tập train qua kiểm định chéo StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xs_train_cv = cross_val_predict(best_model, X_train_encoded, y_train, cv=cv, method='predict_proba')[:, 1]

TARGET_RECALL = 0.70
optimal_threshold = 0.5
best_precision = 0.0

for th in np.arange(0.05, 0.95, 0.01):
    pred_tr = (xs_train_cv >= th).astype(int)
    r = recall_score(y_train, pred_tr, zero_division=0)
    p = precision_score(y_train, pred_tr, zero_division=0)
    if r >= TARGET_RECALL and p > best_precision:
        best_precision = p
        optimal_threshold = round(float(th), 2)

print(f"6. Ngưỡng quyết định tối ưu tìm được trên tập Train: {optimal_threshold} (Mục tiêu Recall >= {TARGET_RECALL})")

# Đánh giá trên tập test
xs_test = best_model.predict_proba(X_test_encoded)[:, 1]
y_pred_test = (xs_test >= optimal_threshold).astype(int)

test_acc = accuracy_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, xs_test)

print("\n=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST VỚI NGƯỠNG TỐI ƯU ===")
print(f"Accuracy : {test_acc:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"F1-Score : {test_f1:.4f}")
print(f"ROC-AUC  : {test_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))

# 11. Lưu Artifacts dưới dạng dictionary chuẩn của scikit-learn & joblib
os.makedirs("model_artifacts", exist_ok=True)

artifacts = {
    "model": best_model,
    "preprocessor": preprocessor,
    "skewed_cols": skewed_cols,
    "optimal_threshold": optimal_threshold,
    "feature_names": list(feature_names),
    "input_columns": list(X.columns)
}

joblib.dump(artifacts, "model_artifacts/attrition_model.joblib")
print("7. Đã lưu artifact chuẩn vào: model_artifacts/attrition_model.joblib")

meta_data = {
    "model_name": "XGBoost Classifier (Attrition Prediction)",
    "optimal_threshold": optimal_threshold,
    "metrics_on_test": {
        "accuracy": round(test_acc, 4),
        "recall": round(test_recall, 4),
        "precision": round(test_precision, 4),
        "f1_score": round(test_f1, 4),
        "roc_auc": round(test_auc, 4)
    },
    "input_features": list(X.columns),
    "skewed_cols_log1p": skewed_cols,
    "encoded_features_count": len(feature_names)
}

with open("model_artifacts/model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta_data, f, indent=2, ensure_ascii=False)
print("8. Đã lưu metadata vào: model_artifacts/model_meta.json")
print("=== HUẤN LUYỆN & XUẤT MODEL HOÀN TẤT ===")
