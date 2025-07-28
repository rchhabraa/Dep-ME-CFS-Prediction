import pandas as pd
import numpy as np
import streamlit as st
import sklearn 
import pickle


print(sklearn.__version__)
def load_model():
    models={
        'LogisticRegression':pickle.load(open('lr_cfs.pkl','rb')),
        'DecisionTree':pickle.load(open('dt_cfs.pkl','rb')),
        'RandomForest':pickle.load(open('rf_cfs.pkl','rb')),
    }
    return models

def main():
    st.title('ME/CFS and Depression Diagnosis Predictor')

    models=load_model()
    st.sidebar.header('Model Selection')
    model_name=st.sidebar.selectbox('Choose a predicted model',
                                    ['LogisticRegression','DecisionTree','RandomForest'])
    model=models.get(model_name)
    st.write(f'Trained model Configuration-{model}')
    if model is None:
        st.error('Selected model not available.')
        return
    
    st.header('Patient Information')

    col1,col2=st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=70, value=30)
        sleep_quality_index = st.slider('Sleep Quality Index (1-10)', 1.0, 10.0,5.0,step=0.1)
        brain_fog_level = st.slider('Brain Fog Level (1-10)', 1.0, 10.0, 5.0,step=0.1)
        physical_pain_score = st.slider('Physical Pain Score (1-10)', 1.0, 10.0,5.0,step=0.1)
        stress_level = st.slider('Stress Level (1-10)', 1.0, 10.0, 3.0,step=0.1)
        depression_phq9_score = st.slider('PHQ-9 Depression Score (0-27)', 0.0, 27.0,10.0,step=0.1)
        fatigue_severity = st.slider('Fatigue Severity Scale Score (1-7)', 1.0, 7.0, 4.0, step=0.1)

    with col2:
        pem_duration = st.number_input('PEM Duration (hours)', min_value=0, max_value=72, value=24)
        hours_of_sleep = st.number_input('Hours of Sleep per Night', min_value=0, max_value=24, value=7)
        pem_present = st.selectbox('PEM Present', ['Yes', 'No'])
        gender = st.selectbox('Gender', ['Male', 'Female'])
        work_status = st.selectbox('Work Status', ['Partially working', 'Working', 'Not working'])
        social_activity = st.selectbox('Social Activity Level', ['Very Low', 'Low', 'Medium','High','Very High'])
        exercise_frequency = st.selectbox('Exercise Frequency', 
                                    ['Often', 'Rarely', 'Sometimes', 'Never', 'Daily'])
        meditation = st.selectbox('Meditation/Mindfulness Practice', ['Yes', 'No'])

    pem_present = 1 if pem_present=='Yes' else 0
    if gender == 'Male':
        gender_male = 1
        gender_female= 0
    else:
        gender_male = 0
        gender_female = 1

    if work_status == "Partially Working":
        ws_pw = 1
        ws_w = 0
        ws_nw = 0
    elif work_status == "Working":
        ws_pw = 0
        ws_w = 1
        ws_nw = 0
    else:
        ws_pw = 0
        ws_w = 0
        ws_nw = 1

    if social_activity == "Low":
        sa_low = 1
        sa_med = 0
        sa_high = 0
        sa_vh = 0
        sa_vl = 0
    elif social_activity == "Medium":
        sa_low = 0
        sa_med = 1
        sa_high = 0
        sa_vh = 0
        sa_vl = 0
    elif social_activity == "High":
        sa_low = 0
        sa_med = 0
        sa_high = 1
        sa_vh = 0
        sa_vl = 0
    elif social_activity == "Very High":
        sa_low = 0
        sa_med = 0
        sa_high = 0
        sa_vh = 1
        sa_vl = 0
    else :
        sa_low = 0
        sa_med = 0
        sa_high = 0
        sa_vh = 0
        sa_vl = 1

    if exercise_frequency == "Often":
        ef_often = 1
        ef_rarely = 0
        ef_sometimes = 0
        ef_never = 0
        ef_daily = 0
    elif exercise_frequency == "Rarely":
        ef_often = 0
        ef_rarely = 1
        ef_sometimes = 0
        ef_never = 0
        ef_daily = 0
    elif exercise_frequency == "Sometimes":
        ef_often = 0
        ef_rarely = 0
        ef_sometimes = 1
        ef_never = 0
        ef_daily = 0
    elif exercise_frequency == "Never":
        ef_often = 0
        ef_rarely = 0
        ef_sometimes = 0
        ef_never = 1
        ef_daily = 0
    else:
        ef_often = 0
        ef_rarely = 0
        ef_sometimes = 0
        ef_never = 0
        ef_daily = 1

    if meditation == "Yes":
        med_yes = 1
        med_no = 0
    else:
        med_yes = 0
        med_no = 1

    input_data = {
        'age' : age,
        'sleep_quality_index':sleep_quality_index,
        'brain_fog_level': brain_fog_level,
        'physical_pain_score': physical_pain_score,
        'stress_level': stress_level,
        'depression_phq9_score' : depression_phq9_score,
        'fatigue_severity_scale_score': fatigue_severity,
        'pem_duration_hours': pem_duration,
        'hours_of_sleep_per_night': hours_of_sleep,
        'pem_present': pem_present,
        'gender_Male':gender_male,
        'work_status_Partially working': ws_pw,
        'work_status_Working':ws_w,
        'social_activity_level_Low':sa_low,
        'social_activity_level_Medium':sa_med,
        'social_activity_level_Very high': sa_vh,
        'social_activity_level_Very low': sa_vl,
        'exercise_frequency_Never': ef_never,
        'exercise_frequency_Often':ef_often,
        'exercise_frequency_Rarely':ef_rarely,
        'exercise_frequency_Sometimes': ef_sometimes, 
        'meditation_or_mindfulness_Yes' : med_yes
    }

    input_df = pd.DataFrame([input_data])
    st.write(input_df)


    if st.button('Predict Diagnosis'):
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)[0]

        prob_df = pd.DataFrame({
            'Categories': ['Both','Depression','ME_CFS'],
            'Probability': prob
        }).sort_values('Probability', ascending=False)

        st.success(f'Predicted Diagnosis: {prediction}')
        st.write(prob_df)
        st.bar_chart(prob_df.set_index('Categories'))

main()