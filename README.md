# 👥 HR Employee Attrition Prediction System

Hệ thống AI phân tích và dự đoán nguy cơ nhân viên nghỉ việc (Employee Attrition) dựa trên tập dữ liệu IBM HR Analytics, sử dụng mô hình **XGBoost Classifier** kết hợp kỹ thuật cân bằng tỷ lệ (**Scale-Pos-Weight Ratio**) và **Tối ưu ngưỡng quyết định (Optimal Decision Threshold)**.

Ứng dụng được xây dựng và triển khai giao diện tương tác trực quan bằng **Streamlit** (hỗ trợ cả **FastAPI REST API**).

---

## 🌟 Tính Năng Nổi Bật

- **Mô hình AI tối ưu:**
  - **Độ chính xác (Accuracy):** `90.48%`
  - **Khả năng phát hiện nhân viên nghỉ việc (Recall):** `72.34%` (Đảm bảo phát hiện phần lớn nhân viên có ý định rời đi)
  - **F1-Score:** `70.83%` | **ROC-AUC:** `94.08%`
- **Giao diện Streamlit hiện đại:**
  - 👤 **Dự đoán đơn lẻ:** Form nhập liệu phân theo từng nhóm (Cá nhân, Công việc, Đãi ngộ, Thâm niên), có sẵn các nút nạp nhanh mẫu nguy cơ cao và mẫu nhân viên ổn định.
  - 📁 **Dự đoán hàng loạt (Batch CSV):** Tải file CSV danh sách nhân viên lên, tự động phân loại rủi ro và xuất báo cáo kết quả kèm nút tải file CSV về.
  - 📈 **Phân tích đặc trưng (Model Insights):** Trực quan hóa các yếu tố hàng đầu dẫn đến quyết định nghỉ việc (Làm thêm giờ - OverTime, Lương tháng - MonthlyIncome, Thâm niên, v.v.).
  - ⚙️ **Thanh trượt điều chỉnh ngưỡng:** Tự do điều chỉnh ngưỡng quyết định theo nhu cầu thực tế của phòng HR.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── model_artifacts/
│   ├── attrition_model.joblib      # Model XGBoost + Preprocessor + Threshold tối ưu
│   └── model_meta.json             # Thông tin tham số & chỉ số đánh giá (Accuracy, Recall, F1...)
├── streamlit_app.py                # Ứng dụng Web Dashboard giao diện Streamlit
├── main.py                         # Backend REST API bằng FastAPI
├── train_and_export.py             # Script tự động huấn luyện & xuất model artifact
├── colap.ipynb                     # Notebook phân tích, thử nghiệm và so sánh mô hình
├── requirements.txt                # Danh sách thư viện Python cần thiết
├── Dockerfile                      # Cấu hình container Docker / Cloud Run
└── README.md                       # Tài liệu hướng dẫn sử dụng và triển khai
```

---

## 🚀 1. Chạy Streamlit Cục Bộ (Local)

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Huấn luyện lại model (nếu cần cập nhật)
```bash
python train_and_export.py
```

### Bước 3: Khởi chạy Streamlit
```bash
streamlit run streamlit_app.py
```
Trình duyệt sẽ tự động mở tại địa chỉ: `http://localhost:8501`

---

## 🌐 2. Hướng Dẫn Push Code Lên GitHub

### Bước 1: Khởi tạo Git và Commit
Mở Terminal tại thư mục dự án và chạy:
```bash
git init
git add .
git commit -m "Deploy HR Attrition Prediction with Streamlit & XGBoost"
git branch -M main
```

### Bước 2: Tạo Repository trên GitHub
1. Truy cập [GitHub](https://github.com/) và đăng nhập tài khoản.
2. Bấm nút **New** (hoặc dấu `+` ở góc phải trên) để tạo repository mới:
   - Tên repository: `HR-Employee-Attrition-Prediction` (hoặc tên tùy chọn)
   - Chọn chế độ: **Public** (để dễ dàng kết nối với Streamlit Cloud miễn phí).
   - **Không** tích chọn *Add a README file* (vì đã có sẵn file README trong máy).
   - Bấm **Create repository**.

### Bước 3: Liên kết và Push lên GitHub
Copy lệnh hiển thị trên GitHub của bạn và chạy trong terminal (thay URL bằng link repo của bạn):
```bash
git remote add origin https://github.com/YOUR_USERNAME/HR-Employee-Attrition-Prediction.git
git push -u origin main
```

---

## ☁️ 3. Deploy Miễn Phí Lên Streamlit Community Cloud (Khuyên Dùng)

Streamlit Community Cloud là nền tảng **miễn phí 100%**, tự động deploy từ GitHub chỉ sau vài cú click:

1. Truy cập [https://share.streamlit.io/](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Bấm nút **"Create app"** (hoặc **"New app"**).
3. Điền các thông tin đơn giản:
   - **Repository:** Chọn repository vừa push (ví dụ: `YOUR_USERNAME/HR-Employee-Attrition-Prediction`).
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Bấm **"Deploy!"**
5. Chờ 1-2 phút để hệ thống tự động cài đặt thư viện từ `requirements.txt`. Khi hoàn thành, ứng dụng của bạn sẽ có một đường link công khai vĩnh viễn (ví dụ: `https://hr-attrition-prediction.streamlit.app`).

---

## 📡 4. Tùy Chọn: Chạy Hoặc Deploy FastAPI

Nếu bạn muốn phục vụ qua REST API cho ứng dụng mobile/web khác:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger UI Documentation: `http://localhost:8000/docs`
- Endpoint dự đoán: `POST /predict`
