# Hướng Dẫn Triển Khai Model Lên FastAPI & Google Cloud Run

Tài liệu này hướng dẫn chi tiết cách chạy ứng dụng **FastAPI** phục vụ mô hình dự đoán nghỉ việc (HR Employee Attrition) ở môi trường cục bộ và cách triển khai không máy chủ (Serverless) lên **Google Cloud Run**.

---

## 📁 1. Cấu Trúc Thư Mục Triển Khai

```text
Quyen-AI/
├── model_artifacts/
│   ├── attrition_model.joblib      # Model XGBoost + Preprocessor + Threshold tối ưu
│   └── model_meta.json             # Thông tin tham số & chỉ số đánh giá (Accuracy, Recall, F1...)
├── main.py                         # Ứng dụng FastAPI (REST API + Web UI Dashboard)
├── train_and_export.py             # Script tự động huấn luyện & xuất model artifact
├── requirements.txt                # Danh sách thư viện Python cần thiết
├── Dockerfile                      # Cấu hình container cho Google Cloud Run
├── .dockerignore                   # Bỏ qua các file không cần thiết khi đóng gói Docker
├── colap.ipynb                     # Notebook phân tích, thử nghiệm và xuất model
└── DEPLOY_GUIDE.md                 # Tài liệu hướng dẫn này
```

---

## 🚀 2. Chạy Ứng Dụng FastAPI Tại Cục Bộ (Local)

### Bước 1: Kích hoạt môi trường và cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Huấn luyện & xuất model (nếu chưa có thư mục `model_artifacts/`)
```bash
python train_and_export.py
```

### Bước 3: Khởi chạy FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Sau khi chạy lệnh trên, bạn có thể truy cập ngay:
* 🌐 **Giao diện Web Dashboard trực quan:** [http://localhost:8000/](http://localhost:8000/)  
  *(Có sẵn form điền thông tin nhân viên, nút điền mẫu nguy cơ cao và thanh đo xác suất rủi ro trực tiếp).*
* 📖 **Tài liệu API Swagger UI tương tác:** [http://localhost:8000/docs](http://localhost:8000/docs)
* 🩺 **Kiểm tra trạng thái Service:** [http://localhost:8000/health](http://localhost:8000/health)

---

## ☁️ 3. Triển Khai Lên Google Cloud Run (Serverless)

Google Cloud Run là nền tảng Serverless container của Google Cloud, tự động co giãn từ 0 đến hàng ngàn requests và chỉ tính tiền khi có lượt truy cập (rất tiết kiệm chi phí).

### Cách 1: Triển khai 1 lệnh duy nhất bằng Google Cloud CLI (Khuyên dùng)

Nếu máy tính của bạn đã cài đặt [Google Cloud SDK (gcloud CLI)](https://cloud.google.com/sdk/docs/install):

```bash
# 1. Đăng nhập vào tài khoản Google Cloud
gcloud auth login

# 2. Chọn Project ID của bạn trên GCP
gcloud config set project YOUR_PROJECT_ID

# 3. Kích hoạt Cloud Run API và Cloud Build API (chỉ cần chạy lần đầu)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# 4. Triển khai trực tiếp toàn bộ source code lên Cloud Run
gcloud run deploy hr-attrition-api \
    --source . \
    --region asia-southeast1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1
```

* Cloud Build sẽ tự động đọc `Dockerfile`, build image và triển khai lên Cloud Run.
* Sau 2-3 phút, terminal sẽ trả về đường dẫn Public URL, ví dụ:  
  `https://hr-attrition-api-xyz-as.a.run.app`

---

### Cách 2: Đóng gói Docker Image thủ công

Nếu bạn muốn build container image tại local và đẩy lên Artifact Registry:

```bash
# 1. Build image cục bộ để kiểm tra
docker build -t hr-attrition-api:v1 .

# 2. Chạy thử container cục bộ
docker run -p 8080:8080 hr-attrition-api:v1
# Truy cập http://localhost:8080

# 3. Tag và đẩy lên Google Artifact Registry
gcloud auth configure-docker asia-southeast1-docker.pkg.dev

docker tag hr-attrition-api:v1 asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/my-repo/hr-attrition-api:v1
docker push asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/my-repo/hr-attrition-api:v1

# 4. Deploy từ Artifact Registry lên Cloud Run
gcloud run deploy hr-attrition-api \
    --image asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/my-repo/hr-attrition-api:v1 \
    --region asia-southeast1 \
    --allow-unauthenticated
```

---

## 📡 4. Hướng Dẫn Gọi API Dự Đoán (`/predict`)

### 1. Gọi bằng lệnh `curl` (Terminal / Postman):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 28,
       "BusinessTravel": "Travel_Frequently",
       "DailyRate": 600,
       "Department": "Sales",
       "DistanceFromHome": 24,
       "Education": 2,
       "EducationField": "Marketing",
       "EnvironmentSatisfaction": 1,
       "Gender": "Female",
       "HourlyRate": 50,
       "JobInvolvement": 2,
       "JobLevel": 1,
       "JobRole": "Sales Representative",
       "JobSatisfaction": 1,
       "MaritalStatus": "Single",
       "MonthlyIncome": 2300,
       "MonthlyRate": 12000,
       "NumCompaniesWorked": 3,
       "OverTime": "Yes",
       "PercentSalaryHike": 12,
       "PerformanceRating": 3,
       "RelationshipSatisfaction": 2,
       "StockOptionLevel": 0,
       "TotalWorkingYears": 2,
       "TrainingTimesLastYear": 2,
       "WorkLifeBalance": 1,
       "YearsAtCompany": 1,
       "YearsInCurrentRole": 1,
       "YearsSinceLastPromotion": 0,
       "YearsWithCurrManager": 0
     }'
```

### 2. Dữ liệu JSON phản hồi:
```json
{
  "attrition_prediction": "Yes",
  "attrition_probability": 0.9811,
  "risk_level": "High",
  "optimal_threshold": 0.48,
  "message": "CẢNH BÁO: Nhân viên có nguy cơ NGHỈ VIỆC cao (Xác suất: 98.1% >= ngưỡng 0.48)."
}
```

---

## 🎯 5. Các Điểm Nổi Bật Của Hệ Thống
1. **Xử lý mất cân bằng lớp:** Sử dụng `scale_pos_weight = ratio` (~5.19) giúp mô hình không bị thiên lệch vào lớp ở lại.
2. **Tối ưu ngưỡng quyết định:** Thay vì dùng ngưỡng 0.5 cố định, hệ thống tự động tìm và áp dụng ngưỡng tối ưu (ví dụ $0.48$) để đạt mức **Recall > 70%** và **Accuracy > 90%** trên tập test.
3. **Pipeline đồng bộ:** Toàn bộ bước log-transform, encode và predict được đóng gói chặt chẽ, người dùng chỉ cần truyền đúng 30 thông tin nhân viên gốc là nhận được ngay kết quả.
