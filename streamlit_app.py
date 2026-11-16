import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Thiết lập trang Streamlit
st.set_page_config(
    page_title="HR Attrition Prediction | Hệ Thống Dự Đoán Nghỉ Việc",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Nạp Model Artifacts với bộ nhớ đệm (Cache)
@st.cache_resource
def load_artifacts():
    model_path = "model_artifacts/attrition_model.joblib"
    meta_path = "model_artifacts/model_meta.json"
    
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy file mô hình tại `{model_path}`. Vui lòng chạy `train_and_export.py` trước!")
        return None, None
        
    artifacts = joblib.load(model_path)
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
    return artifacts, metadata

artifacts, metadata = load_artifacts()

# 2. Hàm dự đoán cho DataFrame
def predict_dataframe(df_input, threshold_override=None):
    if artifacts is None:
        return None
        
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    skewed_cols = artifacts["skewed_cols"]
    default_threshold = artifacts["optimal_threshold"]
    threshold = threshold_override if threshold_override is not None else default_threshold

    df_clean = df_input.copy()
    for col in ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])

    # Biến đổi Log1p các biến lệch phải
    for col in skewed_cols:
        if col in df_clean.columns:
            df_clean[col] = np.log1p(df_clean[col])

    # Mã hóa dữ liệu bằng preprocessor
    encoded = preprocessor.transform(df_clean)
    probs = model.predict_proba(encoded)[:, 1]

    results = []
    for prob in probs:
        is_attrition = bool(prob >= threshold)
        if prob >= 0.70:
            risk_level = "Nguy cơ Rất cao (High)"
            color = "red"
        elif prob >= threshold:
            risk_level = "Nguy cơ Trung bình (Medium)"
            color = "orange"
        else:
            risk_level = "An toàn (Low)"
            color = "green"

        results.append({
            "Attrition": "Yes" if is_attrition else "No",
            "Probability": round(float(prob), 4),
            "Risk_Level": risk_level,
            "Risk_Color": color,
            "Threshold": threshold
        })
    return results

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/912/912318.png", width=90)
    st.title("HR Analytics AI")
    st.caption("Mô hình XGBoost Tối Ưu Tỷ Lệ (Ratio) & Ngưỡng Quyết Định")
    st.markdown("---")
    
    st.subheader("⚙️ Cấu hình Ngưỡng (Threshold)")
    optimal_th = artifacts["optimal_threshold"] if artifacts else 0.48
    use_custom_th = st.toggle("Tùy chỉnh ngưỡng thủ công", value=False)
    if use_custom_th:
        current_threshold = st.slider(
            "Ngưỡng quyết định (Decision Threshold):",
            min_value=0.1, max_value=0.9, value=float(optimal_th), step=0.01,
            help="Hệ thống mặc định tối ưu ở 0.48 để đảm bảo Recall >= 70%."
        )
    else:
        current_threshold = float(optimal_th)
        st.info(f"💡 Đang dùng **Ngưỡng Tối Ưu: {optimal_th:.2f}** (được kiểm định chéo để phát hiện trên 72% nhân viên có ý định nghỉ).")

    st.markdown("---")
    st.subheader("📊 Hiệu Năng Mô Hình")
    if metadata and "metrics_on_test" in metadata:
        m = metadata["metrics_on_test"]
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
        c2.metric("Recall", f"{m['recall']*100:.1f}%")
        c1.metric("Precision", f"{m['precision']*100:.1f}%")
        c2.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    
    st.markdown("---")
    st.markdown("📌 **Ứng dụng phục vụ:** Đánh giá rủi ro nhân sự, hỗ trợ bộ phận HR đưa ra chính sách giữ chân kịp thời.")

# --- MAIN CONTENT ---
st.title("🏢 Hệ Thống Dự Đoán Nguy Cơ Nghỉ Việc Nhân Viên")
st.markdown("Hệ thống Machine Learning phân tích các chỉ số nhân sự để cảnh báo sớm nguy cơ rời bỏ doanh nghiệp.")

tab1, tab2, tab3 = st.tabs([
    "👤 Dự Đoán Cá Nhân (Single Prediction)",
    "📁 Dự Đoán Hàng Loạt (Batch Prediction CSV)",
    "📈 Thông Tin Mô Hình & Đặc Trưng (Model Insights)"
])

# ==========================================
# TAB 1: DỰ ĐOÁN ĐƠN LẺ
# ==========================================
with tab1:
    st.subheader("Nhập Thông Tin Chi Tiết Của Nhân Viên")
    
    # Nút nạp dữ liệu mẫu
    col_sample1, col_sample2, _ = st.columns([1.5, 1.5, 3])
    fill_high_risk = col_sample1.button("🚨 Nạp mẫu: Nguy cơ nghỉ cao")
    fill_low_risk = col_sample2.button("✅ Nạp mẫu: Nhân viên ổn định")
    
    # Thiết lập giá trị mặc định dựa vào nút bấm
    if fill_high_risk:
        d_age, d_gender, d_marital = 26, "Female", "Single"
        d_dept, d_role, d_level = "Sales", "Sales Representative", 1
        d_overtime, d_travel, d_dist = "Yes", "Travel_Frequently", 24
        d_income, d_daily, d_hourly, d_rate = 2300, 600, 50, 12000
        d_stock, d_hike, d_perf = 0, 12, 3
        d_env_sat, d_job_sat, d_rel_sat = 1, 1, 2
        d_worklife, d_involve = 1, 2
        d_tot_yrs, d_comp_yrs, d_role_yrs = 2, 1, 1
        d_promo_yrs, d_mgr_yrs, d_num_comp = 0, 0, 3
        d_training, d_edu, d_edu_field = 2, 2, "Marketing"
    elif fill_low_risk:
        d_age, d_gender, d_marital = 42, "Male", "Married"
        d_dept, d_role, d_level = "Research & Development", "Manager", 4
        d_overtime, d_travel, d_dist = "No", "Travel_Rarely", 3
        d_income, d_daily, d_hourly, d_rate = 12500, 1100, 85, 20000
        d_stock, d_hike, d_perf = 2, 18, 3
        d_env_sat, d_job_sat, d_rel_sat = 4, 4, 4
        d_worklife, d_involve = 3, 3
        d_tot_yrs, d_comp_yrs, d_role_yrs = 18, 12, 8
        d_promo_yrs, d_mgr_yrs, d_num_comp = 2, 7, 1
        d_training, d_edu, d_edu_field = 3, 4, "Life Sciences"
    else:
        d_age, d_gender, d_marital = 34, "Male", "Married"
        d_dept, d_role, d_level = "Research & Development", "Research Scientist", 2
        d_overtime, d_travel, d_dist = "No", "Travel_Rarely", 6
        d_income, d_daily, d_hourly, d_rate = 5200, 850, 65, 14500
        d_stock, d_hike, d_perf = 1, 14, 3
        d_env_sat, d_job_sat, d_rel_sat = 3, 3, 3
        d_worklife, d_involve = 3, 3
        d_tot_yrs, d_comp_yrs, d_role_yrs = 9, 5, 3
        d_promo_yrs, d_mgr_yrs, d_num_comp = 1, 3, 2
        d_training, d_edu, d_edu_field = 3, 3, "Life Sciences"

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 1. Thông Tin Cá Nhân")
            age = st.number_input("Tuổi (Age)", min_value=18, max_value=65, value=d_age)
            gender = st.selectbox("Giới tính (Gender)", ["Male", "Female"], index=["Male", "Female"].index(d_gender))
            marital_status = st.selectbox("Hôn nhân (MaritalStatus)", ["Single", "Married", "Divorced"], index=["Single", "Married", "Divorced"].index(d_marital))
            dist_home = st.number_input("Khoảng cách đi làm (km)", min_value=1, max_value=50, value=d_dist)
            education = st.slider("Học vấn (Education 1-5)", min_value=1, max_value=5, value=d_edu)
            education_field = st.selectbox(
                "Chuyên ngành (EducationField)",
                ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
                index=["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"].index(d_edu_field)
            )

        with col2:
            st.markdown("##### 💼 2. Công Việc & Chức Vụ")
            dept = st.selectbox("Phòng ban (Department)", ["Research & Development", "Sales", "Human Resources"], index=["Research & Development", "Sales", "Human Resources"].index(d_dept))
            roles = [
                "Sales Executive", "Research Scientist", "Laboratory Technician",
                "Manufacturing Director", "Healthcare Representative", "Manager",
                "Sales Representative", "Research Director", "Human Resources"
            ]
            job_role = st.selectbox("Vị trí (JobRole)", roles, index=roles.index(d_role) if d_role in roles else 0)
            job_level = st.slider("Cấp bậc (JobLevel 1-5)", min_value=1, max_value=5, value=d_level)
            overtime = st.selectbox("Làm thêm giờ (OverTime)", ["No", "Yes"], index=["No", "Yes"].index(d_overtime), help="Làm thêm giờ là yếu tố tác động rất mạnh đến nguy cơ nghỉ việc.")
            travel = st.selectbox("Công tác (BusinessTravel)", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(d_travel))
            job_involvement = st.slider("Mức độ gắn kết (JobInvolvement 1-4)", min_value=1, max_value=4, value=d_involve)

        with col3:
            st.markdown("##### 💰 3. Thu Nhập & Mức Độ Hài Lòng")
            monthly_income = st.number_input("Lương tháng ($/tháng)", min_value=1000, max_value=30000, value=d_income, step=500)
            stock_option = st.slider("Cổ phiếu thưởng (Stock 0-3)", min_value=0, max_value=3, value=d_stock)
            job_satisfaction = st.slider("Hài lòng công việc (1-4)", min_value=1, max_value=4, value=d_job_sat)
            env_satisfaction = st.slider("Hài lòng môi trường (1-4)", min_value=1, max_value=4, value=d_env_sat)
            work_life = st.slider("Cân bằng Work-Life (1-4)", min_value=1, max_value=4, value=d_worklife)
            rel_satisfaction = st.slider("Quan hệ đồng nghiệp (1-4)", min_value=1, max_value=4, value=d_rel_sat)

        st.markdown("##### ⏳ 4. Lịch Sử & Thâm Niên")
        c4_1, c4_2, c4_3, c4_4 = st.columns(4)
        total_working_years = c4_1.number_input("Tổng số năm kinh nghiệm", min_value=0, max_value=45, value=d_tot_yrs)
        years_at_company = c4_2.number_input("Số năm ở công ty hiện tại", min_value=0, max_value=45, value=d_comp_yrs)
        years_in_role = c4_3.number_input("Năm ở vị trí hiện tại", min_value=0, max_value=25, value=d_role_yrs)
        years_with_mgr = c4_4.number_input("Năm làm việc với sếp hiện tại", min_value=0, max_value=25, value=d_mgr_yrs)
        
        c4_5, c4_6, c4_7, c4_8 = st.columns(4)
        years_since_promo = c4_5.number_input("Năm kể từ lần thăng chức cuối", min_value=0, max_value=20, value=d_promo_yrs)
        num_companies = c4_6.number_input("Số công ty cũ từng làm", min_value=0, max_value=15, value=d_num_comp)
        training_times = c4_7.number_input("Số lần đào tạo năm ngoái", min_value=0, max_value=10, value=d_training)
        salary_hike = c4_8.number_input("% Tăng lương gần nhất", min_value=5, max_value=30, value=d_hike)

        btn_predict = st.form_submit_button("🔍 Phân Tích & Dự Đoán Ngay", use_container_width=True, type="primary")

    if btn_predict:
        input_data = {
            "Age": age,
            "BusinessTravel": travel,
            "DailyRate": d_daily,
            "Department": dept,
            "DistanceFromHome": dist_home,
            "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": env_satisfaction,
            "Gender": gender,
            "HourlyRate": d_hourly,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": d_rate,
            "NumCompaniesWorked": num_companies,
            "OverTime": overtime,
            "PercentSalaryHike": salary_hike,
            "PerformanceRating": d_perf,
            "RelationshipSatisfaction": rel_satisfaction,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promo,
            "YearsWithCurrManager": years_with_mgr
        }
        
        res = predict_dataframe(pd.DataFrame([input_data]), threshold_override=current_threshold)[0]
        
        st.markdown("---")
        st.subheader("🎯 Kết Quả Phân Tích Nguy Cơ")
        
        rc1, rc2, rc3 = st.columns([1.5, 2, 2])
        is_yes = res["Attrition"] == "Yes"
        
        with rc1:
            if is_yes:
                st.error("### 🚨 DỰ ĐOÁN: NGHỈ VIỆC")
            else:
                st.success("### ✅ DỰ ĐOÁN: Ở LẠI")
            st.markdown(f"**Phân loại rủi ro:** `{res['Risk_Level']}`")
            st.caption(f"Ngưỡng áp dụng: **{current_threshold:.2f}**")

        with rc2:
            prob_pct = res["Probability"] * 100
            st.metric(label="Xác Suất Nghỉ Việc", value=f"{prob_pct:.1f}%")
            st.progress(res["Probability"])
            
        with rc3:
            if is_yes:
                st.warning("⚠️ **Hành động đề xuất (HR Actions):**\n- Phỏng vấn 1-1 thăm dò tâm tư nguyện vọng.\n- Xem xét điều chỉnh lộ trình thăng tiến hoặc khối lượng làm thêm giờ (OverTime).\n- Rà soát mức đãi ngộ so với mặt bằng thị trường.")
            else:
                st.info("👍 **Đánh giá:** Nhân viên có chỉ số gắn kết tốt. Nên tiếp tục duy trì các chính sách khích lệ và công nhận đóng góp.")

# ==========================================
# TAB 2: DỰ ĐOÁN HÀNG LOẠT (CSV)
# ==========================================
with tab2:
    st.subheader("Dự Đoán Cho Danh Sách Nhân Viên (File CSV)")
    st.markdown("Tải lên file danh sách nhân viên để hệ thống tự động quét và phân loại rủi ro cho toàn bộ nhân sự.")
    
    uploaded_file = st.file_uploader("Chọn file CSV:", type=["csv"])
    
    # Nút tải file mẫu nếu người dùng chưa có
    if st.button("📥 Dùng thử file dữ liệu mẫu có sẵn trong hệ thống"):
        if os.path.exists("attritioon.csv"):
            sample_df = pd.read_csv("attritioon.csv").head(25)
            uploaded_file = "sample"
        else:
            st.warning("Không tìm thấy file mẫu.")

    if uploaded_file is not None:
        if uploaded_file == "sample":
            df_batch = pd.read_csv("attritioon.csv").head(25)
        else:
            df_batch = pd.read_csv(uploaded_file)
            
        st.write(f"Đã nạp **{len(df_batch)} nhân viên**. Xem trước dữ liệu:")
        st.dataframe(df_batch.head(5))
        
        if st.button("🚀 Thực Hiện Dự Đoán Toàn Bộ", type="primary"):
            with st.spinner("Đang tính toán dự đoán..."):
                preds = predict_dataframe(df_batch, threshold_override=current_threshold)
                
                df_result = df_batch.copy()
                df_result["Dự_Đoán_Attrition"] = [p["Attrition"] for p in preds]
                df_result["Xác_Suất_Nghỉ"] = [p["Probability"] for p in preds]
                df_result["Mức_Độ_Rủi_Ro"] = [p["Risk_Level"] for p in preds]
                
                # Thống kê tổng quan
                total_emp = len(df_result)
                yes_count = (df_result["Dự_Đoán_Attrition"] == "Yes").sum()
                no_count = total_emp - yes_count
                
                st.markdown("---")
                st.subheader("📊 Thống Kê Tổng Hợp")
                s1, s2, s3 = st.columns(3)
                s1.metric("Tổng Nhân Viên Quét", total_emp)
                s2.metric("Số Lượng Nguy Cơ Nghỉ (Yes)", f"{yes_count} ({yes_count/total_emp*100:.1f}%)")
                s3.metric("Số Lượng Ổn Định (No)", f"{no_count} ({no_count/total_emp*100:.1f}%)")
                
                st.dataframe(df_result[[
                    "Age", "Department", "JobRole", "MonthlyIncome", "OverTime",
                    "Dự_Đoán_Attrition", "Xác_Suất_Nghỉ", "Mức_Độ_Rủi_Ro"
                ]])
                
                # Nút download CSV kết quả
                csv_bytes = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tải Về Kết Quả Dự Đoán (.CSV)",
                    data=csv_bytes,
                    file_name="HR_Attrition_Predictions.csv",
                    mime="text/csv"
                )

# ==========================================
# TAB 3: INSIGHTS & TỐI ƯU MÔ HÌNH
# ==========================================
with tab3:
    st.subheader("💡 Hiểu Về Mô Hình & Các Yếu Tố Rủi Ro Chính")
    
    st.markdown("""
    #### 1. Các Yếu Tố Ảnh Hưởng Lớn Nhất Đến Quyết Định Nghỉ Việc (Feature Importance):
    Qua phân tích thực nghiệm từ mô hình **XGBoost Classifier**, các yếu tố sau có trọng số tác động hàng đầu:
    - ⏱️ **Làm thêm giờ (OverTime):** Nhân viên thường xuyên làm thêm giờ có tỷ lệ nghỉ việc cao gấp **3 lần** so với người không làm thêm.
    - 💵 **Mức thu nhập hàng tháng (MonthlyIncome):** Nhóm thu nhập dưới $3,000/tháng có tỷ lệ biến động cao nhất.
    - 💍 **Tình trạng hôn nhân (MaritalStatus - Single):** Nhóm độc thân có xu hướng dịch chuyển công việc linh hoạt hơn nhóm đã kết hôn.
    - 🏢 **Số năm ở công ty & Số năm làm việc với sếp hiện tại:** Khoảng thời gian từ 1-2 năm đầu là giai đoạn nhạy cảm nhất.
    - 🚗 **Khoảng cách đi làm (DistanceFromHome):** Khoảng cách trên 15km kết hợp giờ làm kéo dài làm giảm cân bằng Work-Life Balance.

    #### 2. Vì Sao Cần Tối Ưu Tỷ Lệ (Ratio) & Ngưỡng Quyết Định (Threshold)?
    - Dữ liệu nhân sự thực tế bị **mất cân bằng lớp nghiêm trọng** (~84% nhân viên ở lại, chỉ 16% nhân viên nghỉ việc).
    - Nếu dùng mô hình thông thường với ngưỡng mặc định 0.5, mô hình sẽ có xu hướng đoán an toàn (đoán tất cả ở lại) $\\to$ **bỏ sót nhân viên sắp nghỉ (Recall rất thấp hoặc bằng 0)**.
    - Bằng cách đặt tham số `scale_pos_weight = ratio` (~5.19) và tìm ngưỡng tối ưu `threshold = 0.48`, hệ thống đạt mức **Recall trên 72%** mà vẫn giữ độ chính xác tổng thể trên **90%**!
    """)
