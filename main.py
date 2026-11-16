import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Đường dẫn file artifact
MODEL_PATH = "model_artifacts/attrition_model.joblib"
META_PATH = "model_artifacts/model_meta.json"

artifacts = None
metadata = {}

def load_artifacts():
    global artifacts, metadata
    if os.path.exists(MODEL_PATH):
        artifacts = joblib.load(MODEL_PATH)
        print(f"Loaded model artifacts from {MODEL_PATH}")
    else:
        print(f"Warning: {MODEL_PATH} not found.")

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

# Nạp model ngay khi khởi động
load_artifacts()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

# Khởi tạo FastAPI
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API dự đoán nguy cơ nhân viên nghỉ việc (Employee Attrition) dựa trên mô hình XGBoost tối ưu hóa theo tỷ lệ (ratio) và ngưỡng quyết định (threshold).",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hàm xử lý dự đoán nội bộ
def predict_dataframe(df_input: pd.DataFrame):
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa được nạp. Hãy chạy train_and_export.py trước.")

    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    skewed_cols = artifacts["skewed_cols"]
    threshold = artifacts["optimal_threshold"]

    df_clean = df_input.copy()
    for col in ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    # Log1p biến lệch
    for col in skewed_cols:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col])

    # Transform
    encoded = preprocessor.transform(df_clean)
    probs = model.predict_proba(encoded)[:, 1]

    results = []
    for prob in probs:
        is_attrition = bool(prob >= threshold)
        if prob >= 0.70:
            risk_level = "High"
        elif prob >= threshold:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        results.append({
            "attrition_prediction": "Yes" if is_attrition else "No",
            "attrition_probability": round(float(prob), 4),
            "risk_level": risk_level,
            "optimal_threshold": threshold
        })
    return results

# Pydantic Schema
class EmployeeInput(BaseModel):
    Age: int = Field(35, description="Tuổi của nhân viên (18-60)", example=35)
    BusinessTravel: str = Field("Travel_Rarely", description="Non-Travel, Travel_Rarely, Travel_Frequently", example="Travel_Rarely")
    DailyRate: int = Field(800, description="Mức lương hàng ngày (100-1500)", example=800)
    Department: str = Field("Research & Development", description="Sales, Research & Development, Human Resources", example="Research & Development")
    DistanceFromHome: int = Field(5, description="Khoảng cách từ nhà đến công ty (km)", example=5)
    Education: int = Field(3, description="Trình độ học vấn (1-5)", example=3)
    EducationField: str = Field("Life Sciences", description="Life Sciences, Medical, Marketing, Technical Degree, Human Resources, Other", example="Life Sciences")
    EnvironmentSatisfaction: int = Field(3, description="Mức độ hài lòng với môi trường (1-4)", example=3)
    Gender: str = Field("Male", description="Male, Female", example="Male")
    HourlyRate: int = Field(65, description="Lương theo giờ", example=65)
    JobInvolvement: int = Field(3, description="Mức độ gắn kết công việc (1-4)", example=3)
    JobLevel: int = Field(2, description="Cấp bậc công việc (1-5)", example=2)
    JobRole: str = Field("Research Scientist", description="Sales Executive, Research Scientist, Laboratory Technician, etc.", example="Research Scientist")
    JobSatisfaction: int = Field(3, description="Độ hài lòng công việc (1-4)", example=3)
    MaritalStatus: str = Field("Married", description="Single, Married, Divorced", example="Married")
    MonthlyIncome: int = Field(5000, description="Thu nhập hàng tháng ($)", example=5000)
    MonthlyRate: int = Field(15000, description="Định mức hàng tháng", example=15000)
    NumCompaniesWorked: int = Field(2, description="Số công ty từng làm việc", example=2)
    OverTime: str = Field("No", description="Làm thêm giờ (Yes/No)", example="No")
    PercentSalaryHike: int = Field(15, description="% tăng lương gần nhất", example=15)
    PerformanceRating: int = Field(3, description="Đánh giá hiệu suất (1-4)", example=3)
    RelationshipSatisfaction: int = Field(3, description="Mối quan hệ đồng nghiệp (1-4)", example=3)
    StockOptionLevel: int = Field(1, description="Cổ phiếu thưởng (0-3)", example=1)
    TotalWorkingYears: int = Field(10, description="Tổng số năm kinh nghiệm", example=10)
    TrainingTimesLastYear: int = Field(3, description="Số lần đào tạo năm ngoái", example=3)
    WorkLifeBalance: int = Field(3, description="Cân bằng công việc & cuộc sống (1-4)", example=3)
    YearsAtCompany: int = Field(5, description="Số năm làm việc tại công ty", example=5)
    YearsInCurrentRole: int = Field(3, description="Số năm ở vị trí hiện tại", example=3)
    YearsSinceLastPromotion: int = Field(1, description="Số năm kể từ lần thăng chức cuối", example=1)
    YearsWithCurrManager: int = Field(3, description="Số năm làm việc với sếp hiện tại", example=3)

class PredictionResponse(BaseModel):
    attrition_prediction: str
    attrition_probability: float
    risk_level: str
    optimal_threshold: float
    message: str

# 1. Health check
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": artifacts is not None,
        "metadata": metadata
    }

# 2. Dự đoán đơn lẻ (Single Prediction)
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_employee(employee: EmployeeInput):
    df_in = pd.DataFrame([employee.dict()])
    res = predict_dataframe(df_in)[0]

    if res["attrition_prediction"] == "Yes":
        msg = f"CẢNH BÁO: Nhân viên có nguy cơ NGHỈ VIỆC cao (Xác suất: {res['attrition_probability']*100:.1f}% >= ngưỡng {res['optimal_threshold']})."
    else:
        msg = f"AN TOÀN: Nhân viên có xu hướng Ở LẠI (Xác suất nghỉ: {res['attrition_probability']*100:.1f}% < ngưỡng {res['optimal_threshold']})."

    return {
        **res,
        "message": msg
    }

# 3. Dự đoán hàng loạt (Batch Prediction)
@app.post("/predict_batch", tags=["Prediction"])
def predict_batch_employees(employees: List[EmployeeInput]):
    df_in = pd.DataFrame([e.dict() for e in employees])
    results = predict_dataframe(df_in)
    return {
        "total_employees": len(results),
        "predictions": results
    }

# 4. Giao diện Web trực quan (Interactive Dashboard UI)
@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def serve_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HR Attrition Prediction Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .card { border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.06); border: none; }
            .header-bar { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 25px 0; margin-bottom: 25px; }
            .btn-predict { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); border: none; padding: 12px 30px; font-weight: bold; border-radius: 25px; }
            .btn-predict:hover { opacity: 0.9; color: white; }
            .result-card { display: none; }
            .risk-High { color: #dc3545; font-weight: bold; }
            .risk-Medium { color: #ffc107; font-weight: bold; }
            .risk-Low { color: #28a745; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header-bar text-center">
            <h2><i class="fa-solid fa-users-gear me-2"></i> Hệ Thống Dự Đoán Nguy Cơ Nghỉ Việc (HR Attrition)</h2>
            <p class="mb-0">FastAPI & Optimized XGBoost (Tối ưu Ngưỡng & Tỷ lệ Mất Cân Bằng)</p>
        </div>

        <div class="container mb-5">
            <div class="row">
                <div class="col-lg-8">
                    <div class="card p-4">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 class="mb-0"><i class="fa-solid fa-id-card me-2 text-primary"></i> Thông Tin Nhân Viên</h5>
                            <button type="button" class="btn btn-sm btn-outline-danger" onclick="fillRiskSample()">
                                <i class="fa-solid fa-triangle-exclamation me-1"></i> Điền Mẫu Nguy Cơ Cao
                            </button>
                        </div>
                        <form id="predictionForm">
                            <div class="row g-3">
                                <div class="col-md-4">
                                    <label class="form-label">Tuổi (Age)</label>
                                    <input type="number" class="form-control" id="Age" value="32" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Giới tính (Gender)</label>
                                    <select class="form-select" id="Gender">
                                        <option value="Male">Male</option>
                                        <option value="Female">Female</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Hôn nhân (MaritalStatus)</label>
                                    <select class="form-select" id="MaritalStatus">
                                        <option value="Single">Single (Độc thân)</option>
                                        <option value="Married" selected>Married (Đã kết hôn)</option>
                                        <option value="Divorced">Divorced (Ly hôn)</option>
                                    </select>
                                </div>

                                <div class="col-md-4">
                                    <label class="form-label fw-bold text-danger">Làm thêm giờ (OverTime)</label>
                                    <select class="form-select border-danger" id="OverTime">
                                        <option value="Yes">Yes (Có)</option>
                                        <option value="No" selected>No (Không)</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Công tác (BusinessTravel)</label>
                                    <select class="form-select" id="BusinessTravel">
                                        <option value="Travel_Rarely" selected>Travel_Rarely</option>
                                        <option value="Travel_Frequently">Travel_Frequently</option>
                                        <option value="Non-Travel">Non-Travel</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Khoảng cách đi làm (km)</label>
                                    <input type="number" class="form-control" id="DistanceFromHome" value="7">
                                </div>

                                <div class="col-md-6">
                                    <label class="form-label">Phòng ban (Department)</label>
                                    <select class="form-select" id="Department">
                                        <option value="Research & Development" selected>Research & Development</option>
                                        <option value="Sales">Sales</option>
                                        <option value="Human Resources">Human Resources</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Vị trí (JobRole)</label>
                                    <select class="form-select" id="JobRole">
                                        <option value="Research Scientist" selected>Research Scientist</option>
                                        <option value="Laboratory Technician">Laboratory Technician</option>
                                        <option value="Sales Executive">Sales Executive</option>
                                        <option value="Manufacturing Director">Manufacturing Director</option>
                                        <option value="Healthcare Representative">Healthcare Representative</option>
                                        <option value="Manager">Manager</option>
                                        <option value="Sales Representative">Sales Representative</option>
                                        <option value="Research Director">Research Director</option>
                                        <option value="Human Resources">Human Resources</option>
                                    </select>
                                </div>

                                <div class="col-md-4">
                                    <label class="form-label">Lương tháng ($)</label>
                                    <input type="number" class="form-control" id="MonthlyIncome" value="4500">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Cấp bậc (JobLevel 1-5)</label>
                                    <input type="number" class="form-control" id="JobLevel" value="2" min="1" max="5">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Cổ phiếu thưởng (Stock 0-3)</label>
                                    <input type="number" class="form-control" id="StockOptionLevel" value="1" min="0" max="3">
                                </div>

                                <div class="col-md-4">
                                    <label class="form-label">Hài lòng công việc (1-4)</label>
                                    <input type="number" class="form-control" id="JobSatisfaction" value="3" min="1" max="4">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Hài lòng môi trường (1-4)</label>
                                    <input type="number" class="form-control" id="EnvironmentSatisfaction" value="3" min="1" max="4">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Cân bằng Work-Life (1-4)</label>
                                    <input type="number" class="form-control" id="WorkLifeBalance" value="3" min="1" max="4">
                                </div>

                                <div class="col-md-4">
                                    <label class="form-label">Tổng năm kinh nghiệm</label>
                                    <input type="number" class="form-control" id="TotalWorkingYears" value="8">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Số năm ở công ty</label>
                                    <input type="number" class="form-control" id="YearsAtCompany" value="4">
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Năm với sếp hiện tại</label>
                                    <input type="number" class="form-control" id="YearsWithCurrManager" value="3">
                                </div>
                            </div>

                            <div class="text-center mt-4">
                                <button type="button" class="btn btn-predict text-white" onclick="makePrediction()">
                                    <i class="fa-solid fa-chart-line me-2"></i> Dự Đoán Ngay
                                </button>
                            </div>
                        </form>
                    </div>
                </div>

                <div class="col-lg-4">
                    <div class="card p-4 text-center" id="emptyResult">
                        <i class="fa-solid fa-chart-pie fa-4x text-muted my-4"></i>
                        <h6 class="text-muted">Nhập thông tin nhân viên và bấm <strong>Dự Đoán Ngay</strong> để xem phân tích rủi ro.</h6>
                    </div>

                    <div class="card p-4 result-card" id="resultCard">
                        <h5><i class="fa-solid fa-clipboard-check text-success me-2"></i> Kết Quả Phân Tích</h5>
                        <hr>
                        <div class="my-3">
                            <span class="text-muted">Dự đoán Attrition:</span>
                            <h2 id="resPrediction" class="my-2"></h2>
                            <span id="resRiskBadge" class="badge p-2"></span>
                        </div>

                        <div class="my-3 text-start">
                            <label class="form-label fw-bold">Xác suất rời đi: <span id="probText"></span></label>
                            <div class="progress" style="height: 24px;">
                                <div id="probBar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar"></div>
                            </div>
                            <small class="text-muted d-block mt-1">Ngưỡng tối ưu hệ thống: <strong id="threshText"></strong></small>
                        </div>

                        <div class="alert alert-secondary mt-3 text-start" id="resMsg" style="font-size: 0.9rem;"></div>
                    </div>

                    <div class="card p-3 mt-3 text-center">
                        <small class="text-muted">Tài liệu API Swagger UI:</small>
                        <a href="/docs" target="_blank" class="btn btn-sm btn-outline-primary mt-2">
                            <i class="fa-solid fa-book-open me-1"></i> Mở API Docs (/docs)
                        </a>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function fillRiskSample() {
                document.getElementById('Age').value = 26;
                document.getElementById('MaritalStatus').value = 'Single';
                document.getElementById('OverTime').value = 'Yes';
                document.getElementById('BusinessTravel').value = 'Travel_Frequently';
                document.getElementById('DistanceFromHome').value = 24;
                document.getElementById('MonthlyIncome').value = 2300;
                document.getElementById('JobLevel').value = 1;
                document.getElementById('StockOptionLevel').value = 0;
                document.getElementById('JobSatisfaction').value = 1;
                document.getElementById('EnvironmentSatisfaction').value = 1;
                document.getElementById('WorkLifeBalance').value = 1;
                document.getElementById('TotalWorkingYears').value = 2;
                document.getElementById('YearsAtCompany').value = 1;
                document.getElementById('YearsWithCurrManager').value = 0;
            }

            async function makePrediction() {
                const payload = {
                    Age: parseInt(document.getElementById('Age').value) || 30,
                    BusinessTravel: document.getElementById('BusinessTravel').value,
                    DailyRate: 800,
                    Department: document.getElementById('Department').value,
                    DistanceFromHome: parseInt(document.getElementById('DistanceFromHome').value) || 5,
                    Education: 3,
                    EducationField: "Life Sciences",
                    EnvironmentSatisfaction: parseInt(document.getElementById('EnvironmentSatisfaction').value) || 3,
                    Gender: document.getElementById('Gender').value,
                    HourlyRate: 65,
                    JobInvolvement: 3,
                    JobLevel: parseInt(document.getElementById('JobLevel').value) || 2,
                    JobRole: document.getElementById('JobRole').value,
                    JobSatisfaction: parseInt(document.getElementById('JobSatisfaction').value) || 3,
                    MaritalStatus: document.getElementById('MaritalStatus').value,
                    MonthlyIncome: parseInt(document.getElementById('MonthlyIncome').value) || 5000,
                    MonthlyRate: 15000,
                    NumCompaniesWorked: 2,
                    OverTime: document.getElementById('OverTime').value,
                    PercentSalaryHike: 14,
                    PerformanceRating: 3,
                    RelationshipSatisfaction: 3,
                    StockOptionLevel: parseInt(document.getElementById('StockOptionLevel').value) || 1,
                    TotalWorkingYears: parseInt(document.getElementById('TotalWorkingYears').value) || 8,
                    TrainingTimesLastYear: 3,
                    WorkLifeBalance: parseInt(document.getElementById('WorkLifeBalance').value) || 3,
                    YearsAtCompany: parseInt(document.getElementById('YearsAtCompany').value) || 5,
                    YearsInCurrentRole: 3,
                    YearsSinceLastPromotion: 1,
                    YearsWithCurrManager: parseInt(document.getElementById('YearsWithCurrManager').value) || 3
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await response.json();

                    document.getElementById('emptyResult').style.display = 'none';
                    document.getElementById('resultCard').style.display = 'block';

                    const isYes = data.attrition_prediction === "Yes";
                    const probPct = (data.attrition_probability * 100).toFixed(1);

                    document.getElementById('resPrediction').innerText = isYes ? "NGHỈ VIỆC (YES)" : "Ở LẠI (NO)";
                    document.getElementById('resPrediction').className = "my-2 " + (isYes ? "text-danger" : "text-success");

                    const badge = document.getElementById('resRiskBadge');
                    badge.innerText = "Mức độ rủi ro: " + data.risk_level;
                    badge.className = "badge p-2 bg-" + (data.risk_level === 'High' ? 'danger' : (data.risk_level === 'Medium' ? 'warning text-dark' : 'success'));

                    document.getElementById('probText').innerText = probPct + "%";
                    const bar = document.getElementById('probBar');
                    bar.style.width = probPct + "%";
                    bar.className = "progress-bar progress-bar-striped progress-bar-animated bg-" + (isYes ? "danger" : "success");

                    document.getElementById('threshText').innerText = (data.optimal_threshold * 100).toFixed(0) + "%";
                    document.getElementById('resMsg').innerText = data.message;
                } catch (err) {
                    alert("Lỗi khi kết nối API: " + err);
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
