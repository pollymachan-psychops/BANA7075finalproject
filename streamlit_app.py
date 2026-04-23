import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="SparkSync", layout="wide")

st.title("SparkSync")
st.subheader("Care Coordination Decision Support Tool")
st.write("Enter intake-time patient information to generate the top 3 recommended programs.")

PRIMARY_DIAGNOSIS_OPTIONS = [
    "schizophrenia",
    "bipolar",
    "depression",
    "anxiety",
    "ptsd",
    "thought_disorder",
    "dual_diagnosis",
]

PRIORITY_1_OPTIONS = [
    "long-term residential",
    "stabilization",
    "medication management",
    "independent living",
    "life skills development",
    "community integration",
]

PRIORITY_2_OPTIONS = [
    "life skills development",
    "vocational support",
    "community integration",
    "medication management",
    "daily living skills",
    "trauma support",
]

TRANSITION_FOCUS_OPTIONS = [
    "transitional living",
    "independent living",
    "community reintegration",
    "stability",
]

COORDINATOR_COMFORT_OPTIONS = [
    "low",
    "medium",
    "high",
]

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Patient Information")
    age = st.number_input("Age", min_value=0, max_value=100, value=24)
    primary_diagnosis = st.selectbox("Primary Diagnosis", PRIMARY_DIAGNOSIS_OPTIONS, index=0)
    priority_1 = st.selectbox("Priority 1", PRIORITY_1_OPTIONS, index=0)

with col2:
    st.markdown("### Care Needs & Constraints")
    priority_2 = st.selectbox("Priority 2", PRIORITY_2_OPTIONS, index=0)
    transition_focus = st.selectbox("Transition Focus", TRANSITION_FOCUS_OPTIONS, index=0)
    coordinator_comfort = st.selectbox("Coordinator Comfort", COORDINATOR_COMFORT_OPTIONS, index=1)

if st.button("Generate Recommendations"):
    payload = {
        "age": age,
        "primary_diagnosis": primary_diagnosis,
        "priority_1": priority_1,
        "priority_2": priority_2,
        "transition_focus": transition_focus,
        "coordinator_comfort": coordinator_comfort,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        recommendations = data.get("recommendations", [])

        if recommendations:
            st.markdown("## Top 3 Recommendations")

            for i, rec in enumerate(recommendations, start=1):
                st.markdown(f"### {i}. {rec['program']}")
                st.write(f"Location: {rec['location']}")
                st.write(f"Type: {rec['program_type']}")
                st.write(f"Population: {rec['primary_population']}")
                st.write(f"Confidence: {rec['confidence'] * 100:.1f}%")

                if "raw_score" in rec:
                    st.write(f"Raw score: {rec['raw_score']:.6f}")

                st.write("---")
        else:
            st.warning("No recommendations were returned.")

    except requests.exceptions.RequestException as e:
        st.error("Error connecting to the API.")
        st.write(str(e))
    except Exception as e:
        st.error("Unexpected error.")
        st.write(str(e))

st.caption("SparkSync is a decision-support tool. Final placement decisions remain with the care coordinator.")
