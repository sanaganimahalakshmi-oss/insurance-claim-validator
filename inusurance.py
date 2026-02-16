import streamlit as st
import json
import re
from typing import TypedDict, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# -------------------------------
# STATE DEFINITION
# -------------------------------

class ClaimState(TypedDict):
    claim_data: Dict[str, Any]
    extracted_data: Dict[str, Any]
    policy_validation: Dict[str, Any]
    fraud_analysis: Dict[str, Any]
    risk_score: int
    final_decision: Dict[str, Any]
    next_step: str


# -------------------------------
# SUPERVISOR AGENT
# Controls which agent runs next
# -------------------------------

def supervisor_agent(state: ClaimState):

    if "extracted_data" not in state:
        return {"next_step": "DocumentProcessor"}

    if "policy_validation" not in state:
        return {"next_step": "PolicyValidator"}

    if "fraud_analysis" not in state:
        return {"next_step": "FraudDetector"}

    if "risk_score" not in state:
        return {"next_step": "RiskAssessor"}

    if "final_decision" not in state:
        return {"next_step": "DecisionAgent"}

    return {"next_step": "FINISH"}


# -------------------------------
# SAFE JSON EXTRACTOR
# -------------------------------

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {}


# -------------------------------
# AGENT 1: DOCUMENT PROCESSOR
# -------------------------------

def document_processing_agent(state: ClaimState):

    prompt = f"""
    Extract structured claim data in JSON format:
    {state['claim_data']}
    """

    response = st.session_state.llm.invoke([HumanMessage(content=prompt)])

    return {"extracted_data": extract_json(response.content)}


# -------------------------------
# AGENT 2: POLICY VALIDATOR
# -------------------------------

def policy_validation_agent(state: ClaimState):

    prompt = f"""
    Validate policy. Return JSON:
    {{
      "is_valid": true/false,
      "reason": "text"
    }}

    Data:
    {state['extracted_data']}
    """

    response = st.session_state.llm.invoke([HumanMessage(content=prompt)])

    return {"policy_validation": extract_json(response.content)}


# -------------------------------
# AGENT 3: FRAUD DETECTOR
# -------------------------------

def fraud_detection_agent(state: ClaimState):

    prompt = f"""
    Perform fraud analysis. Return JSON:
    {{
      "fraud_risk": "Low/Medium/High"
    }}

    Data:
    {state['extracted_data']}
    """

    response = st.session_state.llm.invoke([HumanMessage(content=prompt)])

    return {"fraud_analysis": extract_json(response.content)}


# -------------------------------
# AGENT 4: RISK ASSESSOR (Python Logic)
# -------------------------------

def risk_assessment_agent(state: ClaimState):

    fraud_level = state["fraud_analysis"].get("fraud_risk", "Low")
    claim_amount = state["extracted_data"].get("claim_amount", 0)

    score = 0

    if fraud_level == "High":
        score += 50
    elif fraud_level == "Medium":
        score += 25

    if claim_amount > 50000:
        score += 30

    return {"risk_score": score}


# -------------------------------
# AGENT 5: FINAL DECISION
# -------------------------------

def decision_agent(state: ClaimState):

    policy_valid = state["policy_validation"].get("is_valid", False)
    risk_score = state["risk_score"]

    if not policy_valid:
        decision = "REJECTED"
        reason = "Policy validation failed"

    elif risk_score > 60:
        decision = "MANUAL_REVIEW"
        reason = "High risk score"

    else:
        decision = "APPROVED"
        reason = "Meets criteria"

    return {
        "final_decision": {
            "decision": decision,
            "reason": reason,
            "risk_score": risk_score
        }
    }


# -------------------------------
# BUILD LANGGRAPH WORKFLOW
# -------------------------------

def build_claim_validator():
    workflow = StateGraph(ClaimState)

    workflow.add_node("Supervisor", supervisor_agent)
    workflow.add_node("DocumentProcessor", document_processing_agent)
    workflow.add_node("PolicyValidator", policy_validation_agent)
    workflow.add_node("FraudDetector", fraud_detection_agent)
    workflow.add_node("RiskAssessor", risk_assessment_agent)
    workflow.add_node("DecisionAgent", decision_agent)

    workflow.set_entry_point("Supervisor")

    workflow.add_conditional_edges(
        "Supervisor",
        lambda state: state["next_step"],
        {
            "DocumentProcessor": "DocumentProcessor",
            "PolicyValidator": "PolicyValidator",
            "FraudDetector": "FraudDetector",
            "RiskAssessor": "RiskAssessor",
            "DecisionAgent": "DecisionAgent",
            "FINISH": END
        }
    )

    workflow.add_edge("DocumentProcessor", "Supervisor")
    workflow.add_edge("PolicyValidator", "Supervisor")
    workflow.add_edge("FraudDetector", "Supervisor")
    workflow.add_edge("RiskAssessor", "Supervisor")
    workflow.add_edge("DecisionAgent", "Supervisor")

    return workflow.compile()


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.set_page_config(page_title="AI Insurance Validator", layout="wide")
st.title("🏥 AI-Based Insurance Claim Validator")

api_key = st.text_input("Enter Gemini API Key", type="password")

if api_key:
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )

policy_number = st.text_input("Policy Number")
claim_amount = st.number_input("Claim Amount", min_value=0)
incident_date = st.date_input("Incident Date")
claim_type = st.selectbox("Claim Type", ["Medical", "Accident", "Vehicle", "Property"])
customer_name = st.text_input("Customer Name")


# -------------------------------
# STEP-BY-STEP EXECUTION
# -------------------------------

if st.button("Validate Claim"):

    claim_data = {
        "policy_number": policy_number,
        "claim_amount": claim_amount,
        "incident_date": str(incident_date),
        "claim_type": claim_type,
        "customer_name": customer_name
    }

    app = build_claim_validator()

    st.subheader("🔄 AI Agents Working Step-by-Step")

    progress = st.progress(0)
    status = st.empty()

    step_progress = {
        "DocumentProcessor": 20,
        "PolicyValidator": 40,
        "FraudDetector": 60,
        "RiskAssessor": 80,
        "DecisionAgent": 100
    }

    final_result = None

    for event in app.stream({"claim_data": claim_data}):

        for node_name, node_state in event.items():

            if node_name == "DocumentProcessor":
                status.info("📄 Extracting claim data...")

            elif node_name == "PolicyValidator":
                status.info("📜 Validating policy...")

            elif node_name == "FraudDetector":
                status.info("🕵️ Detecting fraud risk...")

            elif node_name == "RiskAssessor":
                status.info("📊 Calculating risk score...")

            elif node_name == "DecisionAgent":
                status.info("⚖️ Making final decision...")

            if node_name in step_progress:
                progress.progress(step_progress[node_name])

            if "final_decision" in node_state:
                final_result = node_state["final_decision"]

    st.divider()
    st.subheader("🏁 Final Decision")

    st.success(f"Verdict: {final_result['decision']}")
    st.write("Reason:", final_result["reason"])
    st.write("Risk Score:", final_result["risk_score"])