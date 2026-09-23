import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")

st.title("Sleep Quality Analysis & Prediction System")
st.markdown("Predict sleep quality scores (1–10), sleep disorder risk, and personalized recommendations using single input or batch CSV upload.")

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load('sleep_model.pkl')

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    label_encoders = artifacts['label_encoders']
    feature_names = artifacts['feature_names']
except FileNotFoundError:
    st.error("Error: 'sleep_model.pkl' not found. Run 'python train_model.py' first.")
    st.stop()

# Helper function to encode categorical safely with fallback
def safe_encode(encoder, value, default_index=0):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return default_index

# ---------------- SLEEP DISORDER PREDICTION (Rule-Based) ----------------
def predict_sleep_disorder(bmi, systolic_bp, diastolic_bp, stress_level,
                           sleep_duration, physical_activity, heart_rate):
    """
    Rule-based classifier mimicking the patterns in the Sleep Health dataset.
    Returns: 'None', 'Insomnia', or 'Sleep Apnea'
    """
    bmi_norm = str(bmi).strip().lower()

    # Sleep Apnea indicators: high BMI + elevated BP + low sleep duration
    if bmi_norm in ['obese', 'overweight'] and (systolic_bp >= 130 or diastolic_bp >= 85):
        return 'Sleep Apnea'

    # Insomnia indicators: high stress + short sleep + low activity
    if stress_level >= 7 and sleep_duration < 6.5:
        return 'Insomnia'
    if stress_level >= 8 and physical_activity < 30:
        return 'Insomnia'

    # Elevated heart rate + short sleep can also indicate apnea risk
    if heart_rate >= 85 and sleep_duration < 6.0 and bmi_norm in ['obese', 'overweight']:
        return 'Sleep Apnea'

    return 'None'


# ---------------- RECOMMENDATION ENGINE ----------------
def generate_recommendations(score, disorder, sleep_duration, stress_level,
                             physical_activity, daily_steps, bmi,
                             heart_rate, systolic_bp, diastolic_bp):
    """
    Generate a list of personalized recommendations based on the predicted
    quality score, sleep disorder, and user lifestyle metrics.
    """
    recs = []

    # --- Disorder-specific ---
    if disorder == 'Sleep Apnea':
        recs.append("🫁 **Sleep Apnea risk detected** — consult a sleep specialist for a polysomnography (sleep study).")
        recs.append("⚖️ Work on weight reduction; even a 5–10% drop in body weight can dramatically reduce apnea episodes.")
        recs.append("🛏️ Try sleeping on your side (positional therapy) and elevate the head of your bed slightly.")
        recs.append("🚭 Avoid alcohol and sedatives before bedtime — they worsen airway collapse.")

    elif disorder == 'Insomnia':
        recs.append("🌙 **Insomnia risk detected** — establish a strict sleep schedule (same bedtime/wake time daily).")
        recs.append("🧘 Practice relaxation techniques: 4-7-8 breathing, progressive muscle relaxation, or guided meditation.")
        recs.append("📵 Avoid screens 60 minutes before bed; blue light suppresses melatonin production.")
        recs.append("☕ Cut off caffeine after 2 PM and avoid heavy meals within 3 hours of bedtime.")

    else:
        recs.append("✅ No significant sleep disorder pattern detected — keep up your healthy habits.")

    # --- Quality-score specific ---
    if score <= 4:
        recs.append("📉 Your predicted sleep quality is **low**. Prioritize sleep hygiene before anything else.")
    elif score <= 6:
        recs.append("📊 Your sleep quality is **moderate**. Small consistent changes can push it higher.")
    else:
        recs.append("🌟 Excellent sleep quality — maintain your current routine.")

    # --- Metric-specific nudges ---
    if sleep_duration < 7:
        recs.append(f"⏰ Sleep duration ({sleep_duration} hrs) is below the recommended 7–9 hours. Try to add {round(7 - sleep_duration, 1)} more hours.")
    elif sleep_duration > 9:
        recs.append(f"⏰ Sleep duration ({sleep_duration} hrs) is above typical recommendations. Oversleeping can also hurt quality.")

    if stress_level >= 7:
        recs.append("🧠 High stress detected — consider journaling, therapy, or daily mindfulness for 10 minutes.")

    if physical_activity < 30:
        recs.append("🏃 Physical activity is low — aim for at least 30 minutes of moderate exercise daily.")
    else:
        recs.append(f"💪 Great job with {physical_activity} min/day of physical activity — this strongly supports sleep quality.")

    if daily_steps < 5000:
        recs.append(f"🚶 Daily steps ({daily_steps}) are low — a target of 8,000–10,000 steps supports overall sleep health.")

    bmi_norm = str(bmi).strip().lower()
    if bmi_norm in ['obese', 'overweight']:
        recs.append("🥗 BMI is above the healthy range — a balanced diet and regular exercise will improve breathing during sleep.")

    if systolic_bp >= 130 or diastolic_bp >= 85:
        recs.append("💓 Blood pressure is elevated — reduce sodium, manage stress, and monitor BP regularly.")

    if heart_rate >= 85:
        recs.append("❤️ Resting heart rate is on the higher side — cardiovascular exercise can help lower it over time.")

    return recs


# ---------------- Batch Processing Pipeline ----------------
def process_batch(df_input):
    initial_rows = len(df_input)
    df = df_input.copy()

    # Step 1: Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Step 2: Remove exact duplicate rows
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)

    # Step 3: Domain-specific missing value handling
    if 'Sleep Disorder' in df.columns:
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

    # Step 4: Split Blood Pressure
    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].astype(str).str.split('/', expand=True)
        df['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
        df['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
        df.drop(columns=['Blood Pressure'], inplace=True)

    numeric_expected = [
        'Age', 'Sleep Duration', 'Physical Activity Level', 'Physical Activity',
        'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP'
    ]
    for col in numeric_expected:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Physical Activity Level' in df.columns and 'Physical Activity' not in df.columns:
        df.rename(columns={'Physical Activity Level': 'Physical Activity'}, inplace=True)

    cols_to_check = [c for c in feature_names if c in df.columns]

    pre_drop_count = len(df)
    df = df.dropna(subset=cols_to_check)
    null_rows_removed = pre_drop_count - len(df)

    if df.empty:
        raise ValueError("All rows in the uploaded CSV contained missing values or invalid data after cleaning.")

    df_features = df.copy()

    drop_candidates = ['Person ID', 'Sleep Disorder', 'Quality of Sleep']
    df_features.drop(columns=[c for c in drop_candidates if c in df_features.columns],
                     errors='ignore', inplace=True)

    for col, le in label_encoders.items():
        if col in df_features.columns:
            df_features[col] = df_features[col].astype(str).map(
                lambda val: le.transform([val])[0] if val in le.classes_ else 0
            )

    for col in feature_names:
        if col not in df_features.columns:
            df_features[col] = 0

    df_features = df_features[feature_names]

    # Scale & predict quality
    scaled_data = scaler.transform(df_features)
    predictions = model.predict(scaled_data)
    df['Predicted_Quality_of_Sleep'] = predictions

    # Predict sleep disorder row-wise (rule-based)
    def _disorder_row(row):
        return predict_sleep_disorder(
            bmi=row.get('BMI Category', 'Normal'),
            systolic_bp=row.get('Systolic_BP', 120),
            diastolic_bp=row.get('Diastolic_BP', 80),
            stress_level=row.get('Stress Level', 5),
            sleep_duration=row.get('Sleep Duration', 7),
            physical_activity=row.get('Physical Activity', 60),
            heart_rate=row.get('Heart Rate', 72),
        )

    df['Predicted_Sleep_Disorder'] = df.apply(_disorder_row, axis=1)

    # Generate short recommendation string for each row
    def _recs_row(row):
        recs = generate_recommendations(
            score=row['Predicted_Quality_of_Sleep'],
            disorder=row['Predicted_Sleep_Disorder'],
            sleep_duration=row.get('Sleep Duration', 7),
            stress_level=row.get('Stress Level', 5),
            physical_activity=row.get('Physical Activity', 60),
            daily_steps=row.get('Daily Steps', 7000),
            bmi=row.get('BMI Category', 'Normal'),
            heart_rate=row.get('Heart Rate', 72),
            systolic_bp=row.get('Systolic_BP', 120),
            diastolic_bp=row.get('Diastolic_BP', 80),
        )
        # Strip emojis/bold for CSV readability
        clean = [r.replace('**', '').split('—', 1)[-1].strip() for r in recs]
        return " | ".join(clean)

    df['Recommendations'] = df.apply(_recs_row, axis=1)

    cleaning_stats = {
        'initial_rows': initial_rows,
        'duplicates_removed': duplicates_removed,
        'null_rows_removed': null_rows_removed,
        'final_clean_rows': len(df)
    }

    return df, cleaning_stats


# Tabs for Navigation
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV Upload)"])

# ----------------- TAB 1: SINGLE PREDICTION -----------------
with tab1:
    st.subheader("Manual Input")
    col1, col2 = st.columns(2)

    occupation_list = list(label_encoders['Occupation'].classes_) + ["Other"]

    with col1:
        gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
        age = st.slider("Age", 18, 80, 25)

        selected_occupation = st.selectbox("Occupation", occupation_list)
        if selected_occupation == "Other":
            other_occ = st.text_input("Specify your occupation:", placeholder="e.g., Student, Freelancer")

        sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, step=0.1)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

    with col2:
        bmi = st.selectbox("BMI Category", label_encoders['BMI Category'].classes_)
        physical_activity = st.slider("Physical Activity (min/day)", 0, 180, 60)
        heart_rate = st.slider("Heart Rate (bpm)", 50, 110, 72)
        daily_steps = st.number_input("Daily Steps", min_value=1000, max_value=25000,
                                      value=7000, step=500)
        # Removed: Systolic BP and Diastolic BP inputs

    if st.button("Predict Score", type="primary"):
        enc_gender = safe_encode(label_encoders['Gender'], gender)
        enc_occupation = safe_encode(label_encoders['Occupation'], selected_occupation)
        enc_bmi = safe_encode(label_encoders['BMI Category'], bmi)

        # Fixed defaults for model compatibility
        systolic_bp = 120
        diastolic_bp = 80

        single_row = pd.DataFrame([[
            enc_gender, age, enc_occupation, sleep_duration,
            physical_activity, stress_level, enc_bmi, heart_rate,
            daily_steps, systolic_bp, diastolic_bp
        ]], columns=feature_names)

        scaled_row = scaler.transform(single_row)
        score = model.predict(scaled_row)[0]

        # Predict sleep disorder (rule-based)
        disorder = predict_sleep_disorder(
            bmi=bmi,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            stress_level=stress_level,
            sleep_duration=sleep_duration,
            physical_activity=physical_activity,
            heart_rate=heart_rate,
        )

        # Generate recommendations
        recs = generate_recommendations(
            score=score,
            disorder=disorder,
            sleep_duration=sleep_duration,
            stress_level=stress_level,
            physical_activity=physical_activity,
            daily_steps=daily_steps,
            bmi=bmi,
            heart_rate=heart_rate,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
        )

        st.divider()

        # -------- Result Cards --------
        r1, r2 = st.columns(2)
        with r1:
            st.metric("Predicted Sleep Quality", f"{score} / 10")
        with r2:
            st.metric("Predicted Sleep Disorder", disorder)

        # Quality band
        if score >= 8:
            st.success("High Sleep Quality: Optimal physical indicators and healthy routine balance.")
        elif score >= 5:
            st.warning("Moderate Sleep Quality: Elevated stress or irregular sleep duration observed.")
        else:
            st.error("Poor Sleep Quality: Significant physiological strain or deficit detected.")

        # Disorder band
        if disorder == 'None':
            st.info("🟢 No significant sleep disorder pattern detected.")
        elif disorder == 'Insomnia':
            st.warning("🟡 Insomnia pattern detected — difficulty falling/staying asleep is likely.")
        elif disorder == 'Sleep Apnea':
            st.error("🔴 Sleep Apnea pattern detected — breathing interruptions during sleep are likely.")

        # -------- Recommendations --------
        st.subheader("📋 Personalized Recommendations")
        for rec in recs:
            st.markdown(f"- {rec}")


# ----------------- TAB 2: BATCH CSV PREDICTION -----------------
with tab2:
    st.subheader("Batch Prediction with Automated Data Cleaning")
    st.markdown("""
    **Data Quality Assurance Steps Applied:**
    - Removes exact duplicate records.
    - Strips whitespace and normalizes column headers.
    - Resolves `Blood Pressure` syntax into integer features.
    - Removes rows containing null (`NaN`) values in essential feature columns.
    
    **Predictions produced per row:**
    - `Predicted_Quality_of_Sleep` (1–10)
    - `Predicted_Sleep_Disorder` (None / Insomnia / Sleep Apnea)
    - `Recommendations` (personalized action items)
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="csv_uploader")

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Raw Data Preview:")
            st.dataframe(raw_df.head())

            if st.button("Clean Data & Run Predictions", type="primary"):
                with st.spinner("Cleaning records and generating predictions..."):
                    result_df, stats = process_batch(raw_df)

                    st.divider()
                    st.subheader("Data Cleaning Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Rows Uploaded", stats['initial_rows'])
                    c2.metric("Duplicates Removed", stats['duplicates_removed'])
                    c3.metric("Null/Corrupt Rows Dropped", stats['null_rows_removed'])
                    c4.metric("Valid Rows Predicted", stats['final_clean_rows'])

                    st.success(f"Successfully processed {stats['final_clean_rows']} clean records.")

                    # Show disorder distribution
                    st.subheader("Predicted Sleep Disorder Distribution")
                    st.bar_chart(result_df['Predicted_Sleep_Disorder'].value_counts())

                    # Reorder columns: prediction + disorder + recommendations first
                    priority_cols = ['Predicted_Quality_of_Sleep',
                                     'Predicted_Sleep_Disorder',
                                     'Recommendations']
                    other_cols = [c for c in result_df.columns if c not in priority_cols]
                    st.dataframe(result_df[priority_cols + other_cols])

                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Cleaned Predictions as CSV",
                        data=csv_data,
                        file_name="cleaned_sleep_quality_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as err:
            st.error(f"Processing Error: {err}")