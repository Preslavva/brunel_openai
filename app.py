import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Brunel Recruitment Analytics",
    page_icon="🚀",
    layout="wide"
)

load_dotenv()

def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")

api_key = get_api_key()

if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to .env locally or Streamlit secrets when deployed.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- 3. CSS STYLING ---
brunel_css = """
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #2D2D2D;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #2D2D2D;
        font-weight: 700;
        border: none;
    }
    div.stButton > button {
        background-color: #FCEE21; 
        color: #2D2D2D;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 16px;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #E6D81E;
        color: #000000;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2D2D2D;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(brunel_css, unsafe_allow_html=True)

# --- 4. DATA LOADING ---
@st.cache_data
def load_and_process_data():
    try:
        indeed_df = pd.read_csv("./data/cleaned/indeed.csv")
        linkedin_df = pd.read_csv("./data/cleaned/linkedin.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    # --- Preprocessing Indeed ---
    indeed = indeed_df.copy()
    indeed = indeed.rename(columns={
        "Job post": "job_title",
        "Impressions (sponsored)": "impressions",
        "Clicks (sponsored)": "clicks",
        "Started applications (sponsored)": "applications",
        "Total cost": "cost",
        "Cost per started application": "cost_per_application",
        "Cost per click": "cpc",
        "Click through rate (sponsored)": "ctr",
        "Application start rate (sponsored)": "application_start_rate"
    })

    cols_metrics = [
        "impressions", "clicks", "applications", "cost",
        "cost_per_application", "cpc", "ctr", "application_start_rate"
    ]

    for c in cols_metrics:
        if c in indeed.columns:
            indeed[c] = pd.to_numeric(indeed[c], errors="coerce").fillna(0)

    indeed_prep = indeed.rename(columns={
        "impressions": "indeed_impressions",
        "clicks": "indeed_clicks",
        "applications": "indeed_applications",
        "cost": "indeed_cost",
        "cost_per_application": "indeed_cost_per_application",
        "cpc": "indeed_cpc",
        "ctr": "indeed_ctr",
        "application_start_rate": "indeed_application_start_rate"
    })

    # --- Preprocessing LinkedIn ---
    linkedin = linkedin_df.copy()
    linkedin = linkedin.rename(columns={
        "Jobtitle": "job_title",
        "Total amount of views": "impressions",
        "Unique visits": "clicks",
        "Total number of application clicks": "applications",
        "Cost per application": "cost_per_application",
        "Cost per click": "cpc",
        "Location": "location",
        "Country": "country",
        "Function": "function"
    })

    numeric_cols = ["impressions", "clicks", "applications", "cost_per_application", "cpc"]
    for c in numeric_cols:
        if c in linkedin.columns:
            linkedin[c] = pd.to_numeric(linkedin[c], errors="coerce").fillna(0)

    linkedin["cost"] = linkedin["cost_per_application"] * linkedin["applications"]

    linkedin["ctr"] = np.where(
        linkedin["impressions"] > 0,
        (linkedin["clicks"] / linkedin["impressions"]) * 100,
        0
    )

    linkedin["application_start_rate"] = np.where(
        linkedin["clicks"] > 0,
        (linkedin["applications"] / linkedin["clicks"]) * 100,
        0
    )

    linkedin_prep = linkedin.rename(columns={
        "impressions": "linkedin_impressions",
        "clicks": "linkedin_clicks",
        "applications": "linkedin_applications",
        "cost": "linkedin_cost",
        "cost_per_application": "linkedin_cost_per_application",
        "cpc": "linkedin_cpc",
        "ctr": "linkedin_ctr",
        "application_start_rate": "linkedin_application_start_rate"
    })

    # --- Grouping ---
    i_g = indeed_prep.groupby("job_title").agg({
        "indeed_impressions": "sum",
        "indeed_clicks": "sum",
        "indeed_applications": "sum",
        "indeed_cost": "sum",
        "indeed_cost_per_application": "mean",
        "indeed_cpc": "mean",
        "indeed_ctr": "mean",
        "indeed_application_start_rate": "mean"
    }).reset_index()

    l_g_metrics = linkedin_prep.groupby("job_title").agg({
        "linkedin_impressions": "sum",
        "linkedin_clicks": "sum",
        "linkedin_applications": "sum",
        "linkedin_cost": "sum",
        "linkedin_cost_per_application": "mean",
        "linkedin_cpc": "mean",
        "linkedin_ctr": "mean",
        "linkedin_application_start_rate": "mean"
    }).reset_index()

    context_cols = [c for c in ["job_title", "country", "location", "function"] if c in linkedin_prep.columns]
    l_context = linkedin_prep[context_cols].drop_duplicates("job_title")

    l_g = pd.merge(l_g_metrics, l_context, on="job_title", how="left")

    return pd.merge(i_g, l_g, on="job_title", how="outer")


# --- 5. UTILS & PROMPT ---
def format_value(v, ndigits=2):
    if pd.isna(v):
        return "N/A"
    try:
        return f"{v:.{ndigits}f}"
    except Exception:
        return str(v)


def build_prompt(row):
    job_title = row.get("job_title", "Unknown job title")

    i_impr = format_value(row.get("indeed_impressions"), 0)
    i_clicks = format_value(row.get("indeed_clicks"), 0)
    i_apps = format_value(row.get("indeed_applications"), 0)
    i_ctr = format_value(row.get("indeed_ctr"), 2)
    i_cpc = format_value(row.get("indeed_cpc"), 3)
    i_app_rate = format_value(row.get("indeed_application_start_rate"), 2)
    i_cpa = format_value(row.get("indeed_cost_per_application"), 2)
    i_cost = format_value(row.get("indeed_cost"), 2)

    l_impr = format_value(row.get("linkedin_impressions"), 0)
    l_clicks = format_value(row.get("linkedin_clicks"), 0)
    l_apps = format_value(row.get("linkedin_applications"), 0)
    l_ctr = format_value(row.get("linkedin_ctr"), 2)
    l_cpc = format_value(row.get("linkedin_cpc"), 3)
    l_app_rate = format_value(row.get("linkedin_application_start_rate"), 2)
    l_cpa = format_value(row.get("linkedin_cost_per_application"), 2)
    l_cost = format_value(row.get("linkedin_cost"), 2)

    country = row.get("country", None)
    function = row.get("function", None)
    location = row.get("location", None)

    context_lines = []
    if pd.notna(country):
        context_lines.append(f"- Country: {country}")
    if pd.notna(location):
        context_lines.append(f"- Location: {location}")
    if pd.notna(function):
        context_lines.append(f"- Function: {function}")

    context_block = "\n".join(context_lines) if context_lines else "No additional context provided."

    prompt = f"""
You are an expert in recruitment marketing analytics.

Your goal is to compare the performance of Indeed and LinkedIn for the following job title and decide which platform is better overall for this role.

Job title: {job_title}
Context:
{context_block}

PRIMARY OBJECTIVES:
1. Volume: impressions, clicks, started applications.
2. Efficiency: CPA, CPC, total cost.
3. Final recommendation: Indeed, LinkedIn, or Mixed.

DATA:

Indeed:
- Impressions: {i_impr}
- Clicks: {i_clicks}
- Applications: {i_apps}
- CTR: {i_ctr}
- CPC: {i_cpc}
- Application Start Rate: {i_app_rate}
- CPA: {i_cpa}
- Total Cost: {i_cost}

LinkedIn:
- Impressions: {l_impr}
- Clicks: {l_clicks}
- Applications: {l_apps}
- CTR: {l_ctr}
- CPC: {l_cpc}
- Application Start Rate: {l_app_rate}
- CPA: {l_cpa}
- Total Cost: {l_cost}

INSTRUCTIONS:
- Compare both platforms clearly.
- Balance volume and efficiency.
- If one platform gives more applications but is much more expensive, explain that trade-off.
- Give a final recommendation: Indeed, LinkedIn, or Mixed.
- Keep the answer concise and practical.
- Format the answer in short sections:
  1. Summary
  2. Key Comparison
  3. Recommendation
"""
    return prompt.strip()


def call_openai_analysis(prompt):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.1
        )

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        text_parts = []
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content_item in item.content:
                        text_value = getattr(content_item, "text", None)
                        if text_value:
                            text_parts.append(text_value)

        if text_parts:
            return "".join(text_parts).strip()

        return "No response was returned by the model."

    except Exception as e:
        return f"OpenAI API error: {e}"


# --- 6. MAIN APP UI ---
st.markdown(
    "<h1 style='border-bottom: 4px solid #FCEE21; display: inline-block; padding-bottom: 5px;'>Brunel</h1>",
    unsafe_allow_html=True
)
st.subheader("Recruitment Channel Analytics (Deep Analysis)")

df = load_and_process_data()

if not df.empty:
    job_list = sorted(df["job_title"].astype(str).unique())
    col_sel, col_btn = st.columns([3, 1])

    with col_sel:
        selected_job = st.selectbox("Select Job Title", options=job_list, label_visibility="collapsed")

    with col_btn:
        run_analysis = st.button("ANALYZE REPORT")

    if run_analysis:
        row = df[df["job_title"] == selected_job].iloc[0]

        with st.spinner("Analyzing data via AI..."):
            prompt_text = build_prompt(row)
            ai_result = call_openai_analysis(prompt_text)

        st.success("Analysis Complete")
        st.markdown(ai_result)

else:
    st.warning("Data not loaded. Please ensure 'data/cleaned/indeed.csv' and 'linkedin.csv' exist in the repository.")