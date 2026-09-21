import json
import os
from datetime import date

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Policy-Aware Claim Decision Engine",
    page_icon="🏥",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #777;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 650;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def call_api(payload: dict) -> dict | None:
    """Send a claim payload to FastAPI and return the JSON result."""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json=payload,
            timeout=120,
        )

        if response.status_code != 200:
            st.error(f"API Error: HTTP {response.status_code}")
            st.code(response.text)
            return None

        return response.json()

    except requests.exceptions.Timeout:
        st.error(
            "The API request timed out. "
            "Make sure FastAPI is running and try again."
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI. Start it with:\n\n"
            "uvicorn src.api:app --reload"
        )
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

    return None


def display_result(result: dict) -> None:
    """Render the structured claim-analysis response."""
    st.header("Analysis Result")

    decision = result.get("decision", "UNKNOWN")
    confidence = result.get("confidence", 0)
    validation_status = result.get("validation_status", "UNKNOWN")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Decision", decision)

    with col2:
        try:
            st.metric("Confidence", f"{float(confidence):.2f}")
        except (TypeError, ValueError):
            st.metric("Confidence", str(confidence))

    with col3:
        st.metric("Validation", validation_status)

    missing_evidence = result.get("missing_evidence", [])

    st.subheader("🔎 Missing Evidence")

    if missing_evidence:
        for item in missing_evidence:
            st.warning(item)
    else:
        st.success("No required evidence is currently missing.")

    unsupported_claims = result.get("unsupported_claims", [])

    st.subheader("⚠️ Unsupported Claims")

    if unsupported_claims:
        for item in unsupported_claims:
            st.warning(item)
    else:
        st.success("No unsupported claims detected.")

    key_findings = result.get("key_findings", [])

    st.subheader("📌 Key Findings")

    if key_findings:
        for finding in key_findings:
            st.write(f"• {finding}")
    else:
        st.write("No key findings returned.")

    coverage_findings = result.get("coverage_findings", [])

    with st.expander("✅ Coverage Findings", expanded=True):
        if coverage_findings:
            for finding in coverage_findings:
                st.json(finding)
        else:
            st.write("No coverage findings.")

    exclusion_findings = result.get("exclusion_findings", [])

    with st.expander("🚫 Exclusion Findings", expanded=False):
        if exclusion_findings:
            for finding in exclusion_findings:
                st.json(finding)
        else:
            st.write("No exclusion findings.")

    limit_findings = result.get("limit_findings", [])

    with st.expander("💰 Limit Findings", expanded=True):
        if limit_findings:
            for finding in limit_findings:
                st.json(finding)
        else:
            st.write("No applicable limit findings.")

    expense_calculation = result.get("expense_calculation", {})

    with st.expander("🧮 Expense Calculation", expanded=True):
        if expense_calculation:
            st.json(expense_calculation)
        else:
            st.write("No expense calculation returned.")

    citations = result.get("citations", [])

    with st.expander("📚 Policy Citations", expanded=True):
        if citations:
            for index, citation in enumerate(citations, start=1):
                st.markdown(f"**Citation {index}**")
                st.json(citation)
        else:
            st.write("No citations returned.")

    with st.expander("🛡️ Validation Result", expanded=False):
        st.json(
            {
                "validation_status": result.get("validation_status"),
                "unsupported_claims": result.get(
                    "unsupported_claims", []
                ),
            }
        )

    trace = result.get("trace", [])

    with st.expander("🤖 Multi-Agent Trace", expanded=True):
        if trace:
            for step in trace:
                st.json(step)
        else:
            st.write("No agent trace returned.")

    investigation_plan = result.get("investigation_plan", [])

    with st.expander("🔍 Investigation Plan", expanded=False):
        if investigation_plan:
            for item in investigation_plan:
                st.write(f"• {item}")
        else:
            st.write("No investigation plan returned.")

    with st.expander("📦 Raw API Response", expanded=False):
        st.json(result)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏥 Policy-Aware Multi-Agent RAG Claim Decision Engine</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Analyze synthetic health-insurance claims using policy-grounded "
    "multi-agent reasoning and evidence validation."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("System")
    st.write("**FastAPI:**", API_URL)

    try:
        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if health_response.status_code == 200:
            st.success("API is running")
        else:
            st.error("API returned an error")

    except requests.exceptions.RequestException:
        st.error("FastAPI is not running")

    st.divider()

    st.write("### Workflow")

    st.write(
        """
        1. Case Analysis Agent
        2. Policy Evidence Agent
        3. Coverage & Exclusion Agent
        4. Decision Agent
        5. Validation Agent
        """
    )


# ============================================================
# INPUT MODE
# ============================================================

st.header("Claim Input")

input_mode = st.radio(
    "Choose how to provide the claim:",
    ["Manual Form", "Paste / Upload JSON"],
    horizontal=True,
)


# ============================================================
# JSON INPUT MODE
# ============================================================

if input_mode == "Paste / Upload JSON":

    st.info(
        "Provide a complete claim object matching the /analyze API request schema. "
        "You can paste JSON directly or upload a .json file."
    )

    json_tab_1, json_tab_2 = st.tabs(
        ["📋 Paste JSON", "📁 Upload JSON"]
    )

    default_json = {
        "case_id": "TEST-ABSTENTION-001",
        "policy_id": "POLICY-001",
        "task": "Evaluate hospitalization admissibility and applicable limits",
        "policy_start_date": "2024-01-01",
        "claim_date": "2026-07-03",
        "coverage_months": 30,
        "sum_insured": 1000000,
        "prior_insurer_continuous_years": 0,
        "claimant": {
            "age": 35
        },
        "hospital": {
            "name": "Test Hospital",
            "network": False,
            "registered": None
        },
        "treatment": {
            "type": "Inpatient",
            "admission_hours": 96,
            "diagnosis": "Acute infection",
            "procedure": "Inpatient treatment",
            "pre_existing": False,
            "experimental": False,
            "medical_necessity_confirmed": None
        },
        "expenses": {
            "room_inr": 0,
            "doctor_inr": 0,
            "medicines_diagnostics_inr": 0,
            "pre_claimed_inr": 0,
            "post_claimed_inr": 0,
            "ambulance_inr": 0
        },
        "evidence_context": {
            "hospital_registered": None,
            "medical_necessity_confirmed": None
        },
        "documents": [
            "claim_form",
            "discharge_summary"
        ]
    }

    with json_tab_1:
        json_text = st.text_area(
            "Paste claim JSON",
            value=json.dumps(
                default_json,
                indent=2
            ),
            height=500,
        )

        if st.button(
            "🔍 Analyze JSON Claim",
            type="primary",
            use_container_width=True,
        ):
            try:
                payload = json.loads(json_text)

                if not isinstance(payload, dict):
                    st.error("The JSON must contain an object at the top level.")
                else:
                    result = call_api(payload)

                    if result is not None:
                        display_result(result)

            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")

    with json_tab_2:
        uploaded_file = st.file_uploader(
            "Upload a claim JSON file",
            type=["json"],
            help="Upload a JSON object matching the POST /analyze request schema.",
        )

        if uploaded_file is not None:
            try:
                payload = json.load(uploaded_file)

                if not isinstance(payload, dict):
                    st.error(
                        "The uploaded JSON must contain an object "
                        "at the top level."
                    )
                else:
                    st.success("JSON file loaded successfully.")
                    st.json(payload)

                    if st.button(
                        "🔍 Analyze Uploaded Claim",
                        type="primary",
                        use_container_width=True,
                    ):
                        result = call_api(payload)

                        if result is not None:
                            display_result(result)

            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON file: {exc}")


# ============================================================
# MANUAL INPUT MODE
# ============================================================

else:

    st.subheader("Claim Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        case_id = st.text_input(
            "Case ID",
            value="TEST-001",
        )

    with col2:
        policy_id = st.text_input(
            "Policy ID",
            value="POLICY-001",
        )

    with col3:
        task = st.text_input(
            "Analysis Task",
            value="Evaluate hospitalization admissibility and applicable limits",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        policy_start_date = st.date_input(
            "Policy Start Date",
            value=date(2024, 1, 1),
        )

    with col2:
        claim_date = st.date_input(
            "Claim Date",
            value=date(2026, 7, 3),
        )

    with col3:
        coverage_months = st.number_input(
            "Coverage Months",
            min_value=0,
            value=30,
            step=1,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        sum_insured = st.number_input(
            "Sum Insured (₹)",
            min_value=0.0,
            value=1000000.0,
            step=10000.0,
        )

    with col2:
        prior_insurer_years = st.number_input(
            "Prior Insurer Continuous Years",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

    with col3:
        patient_age = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120,
            value=35,
            step=1,
        )

    st.markdown(
        '<div class="section-title">🏨 Hospital Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        hospital_name = st.text_input(
            "Hospital Name",
            value="Test Hospital",
        )

    with col2:
        network_status = st.selectbox(
            "Network Status",
            options=[
                "Network",
                "Non-Network",
                "Unknown",
            ],
            index=0,
        )

    with col3:
        hospital_registered = st.selectbox(
            "Hospital Registered / Eligible",
            options=[
                "Yes",
                "No",
                "Unknown",
            ],
            index=0,
        )

    st.markdown(
        '<div class="section-title">🩺 Treatment Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        treatment_type = st.selectbox(
            "Treatment Type",
            options=[
                "Inpatient",
                "Day Care",
                "Domiciliary",
                "Outpatient",
                "Unknown",
            ],
            index=0,
        )

    with col2:
        admission_hours = st.number_input(
            "Admission Hours",
            min_value=0.0,
            value=96.0,
            step=1.0,
        )

    with col3:
        diagnosis = st.text_input(
            "Diagnosis",
            value="Acute infection",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        procedure = st.text_input(
            "Procedure / Treatment",
            value="Inpatient treatment",
        )

    with col2:
        pre_existing = st.selectbox(
            "Pre-existing Disease",
            options=[
                "No",
                "Yes",
                "Unknown",
            ],
            index=0,
        )

    with col3:
        experimental = st.selectbox(
            "Experimental / Unproven Treatment",
            options=[
                "No",
                "Yes",
                "Unknown",
            ],
            index=0,
        )

    medical_necessity = st.selectbox(
        "Medical Necessity Confirmed",
        options=[
            "Yes",
            "No",
            "Unknown",
        ],
        index=0,
    )

    st.markdown(
        '<div class="section-title">💰 Claim Expenses</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        room_expense = st.number_input(
            "Room Expense (₹)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
        )

    with col2:
        doctor_expense = st.number_input(
            "Doctor / Professional Expense (₹)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
        )

    with col3:
        medicines_expense = st.number_input(
            "Medicines / Diagnostics (₹)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        pre_hospitalization_expense = st.number_input(
            "Pre-hospitalization Expense (₹)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
        )

    with col2:
        post_hospitalization_expense = st.number_input(
            "Post-hospitalization Expense (₹)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
        )

    with col3:
        ambulance_expense = st.number_input(
            "Ambulance Expense (₹)",
            min_value=0.0,
            value=0.0,
            step=100.0,
        )

    st.markdown(
        '<div class="section-title">📄 Evidence Context</div>',
        unsafe_allow_html=True,
    )

    documents = st.multiselect(
        "Available Documents",
        options=[
            "claim_form",
            "discharge_summary",
            "hospital_registration",
            "medical_necessity_certificate",
            "itemized_bill",
            "claim_history",
            "prior_policy",
        ],
        default=[
            "claim_form",
            "discharge_summary",
        ],
    )

    st.divider()

    analyze_button = st.button(
        "🔍 Analyze Claim",
        type="primary",
        use_container_width=True,
    )

    if analyze_button:

        hospital_registered_value = {
            "Yes": True,
            "No": False,
            "Unknown": None,
        }[hospital_registered]

        medical_necessity_value = {
            "Yes": True,
            "No": False,
            "Unknown": None,
        }[medical_necessity]

        pre_existing_value = {
            "Yes": True,
            "No": False,
            "Unknown": None,
        }[pre_existing]

        experimental_value = {
            "Yes": True,
            "No": False,
            "Unknown": None,
        }[experimental]

        network_value = {
            "Network": True,
            "Non-Network": False,
            "Unknown": None,
        }[network_status]

        claim_payload = {
            "case_id": case_id,
            "policy_id": policy_id,
            "task": task,
            "policy_start_date": policy_start_date.isoformat(),
            "claim_date": claim_date.isoformat(),
            "coverage_months": int(coverage_months),
            "sum_insured": float(sum_insured),
            "prior_insurer_continuous_years": float(
                prior_insurer_years
            ),
            "claimant": {
                "age": int(patient_age),
            },
            "hospital": {
                "name": hospital_name,
                "network": network_value,
                "registered": hospital_registered_value,
            },
            "treatment": {
                "type": treatment_type,
                "admission_hours": float(admission_hours),
                "diagnosis": diagnosis,
                "procedure": procedure,
                "pre_existing": pre_existing_value,
                "experimental": experimental_value,
                "medical_necessity_confirmed": medical_necessity_value,
            },
            "expenses": {
                "room_inr": float(room_expense),
                "doctor_inr": float(doctor_expense),
                "medicines_diagnostics_inr": float(
                    medicines_expense
                ),
                "pre_claimed_inr": float(
                    pre_hospitalization_expense
                ),
                "post_claimed_inr": float(
                    post_hospitalization_expense
                ),
                "ambulance_inr": float(
                    ambulance_expense
                ),
            },
            "evidence_context": {
                "hospital_registered": hospital_registered_value,
                "medical_necessity_confirmed": medical_necessity_value,
            },
            "documents": documents,
        }

        with st.spinner(
            "Running multi-agent claim analysis..."
        ):
            result = call_api(claim_payload)

        if result is not None:
            display_result(result)
