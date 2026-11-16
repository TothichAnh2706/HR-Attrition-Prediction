# 👥 HR Employee Attrition Prediction System

Hệ thống AI phân tích và dự đoán nguy cơ nhân viên nghỉ việc (Employee Attrition) dựa trên tập dữ liệu IBM HR Analytics, sử dụng mô hình **XGBoost Classifier** kết hợp kỹ thuật cân bằng tỷ lệ (**Scale-Pos-Weight Ratio**) và **Tối ưu ngưỡng quyết định (Optimal Decision Threshold)**.

Ứng dụng được xây dựng và triển khai giao diện tương tác trực quan bằng **Streamlit** (hỗ trợ cả **FastAPI REST API**).

---

##  Tính Năng Nổi Bật

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
├── requirements.txt                # Danh sách thư viện Python cần thiết
├── Dockerfile                      # Cấu hình container Docker / Cloud Run
└── README.md                       # Tài liệu hướng dẫn sử dụng và triển khai
```

