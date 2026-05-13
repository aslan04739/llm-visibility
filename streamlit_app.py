import streamlit as st
import json
import pandas as pd
from llm_audit import GRANULAR_PROMPTS, run_granular_audit, generate_visuals_return_figs

st.set_page_config(page_title="LLM Visibility Audit", layout="wide")

st.title("LLM Visibility — Audit 360 (Streamlit)")

with st.sidebar.form(key='keys_and_prompts'):
    st.header("API Keys & Prompts")
    openai_key = st.text_input("OpenAI API key (optional)", type="password")
    gemini_key = st.text_input("Gemini API key (optional)", type="password")
    perplexity_key = st.text_input("Perplexity API key (optional)", type="password")

    use_defaults = st.checkbox("Use default prompts", value=True)
    st.write("Or paste a JSON mapping of name->prompt (example below)")
    sample_json = json.dumps(GRANULAR_PROMPTS, ensure_ascii=False, indent=2)
    prompts_text = st.text_area("Prompts JSON", value=sample_json if use_defaults else "", height=240)

    submit = st.form_submit_button("Save inputs")

if submit:
    st.success("Inputs saved in session — press 'Run audit' to start.")

st.markdown("---")

run_btn = st.button("Run audit and generate graphs")

if run_btn:
    # parse prompts
    try:
        prompts = json.loads(prompts_text)
        if not isinstance(prompts, dict):
            st.error("Prompts JSON should be an object mapping names to prompt strings.")
            st.stop()
    except Exception as e:
        st.error(f"Invalid JSON for prompts: {e}")
        st.stop()

    with st.spinner("Running audit on the selected prompts (this will call APIs)..."):
        df = run_granular_audit(prompts, openai_key=openai_key or None, gemini_key=gemini_key or None, perplexity_key=perplexity_key or None)

    st.success("Audit completed — generating visualizations")

    figs = generate_visuals_return_figs(df)

    col1, col2 = st.columns(2)
    if 'barplot' in figs:
        with col1:
            st.header("Barplot — Scores par Thématique")
            st.pyplot(figs['barplot'])
    if 'consensus' in figs:
        with col2:
            st.header("Nuage de consensus (stripplot)")
            st.pyplot(figs['consensus'])

    # Radar in full width below
    if 'radar' in figs:
        st.header("Radar 360")
        st.pyplot(figs['radar'])

    # Dataframe and CSV download
    st.markdown("---")
    st.header("Raw results")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download results as CSV", data=csv, file_name='audit_360_results.csv', mime='text/csv')

    st.info("Notes: Provide API keys in the sidebar. If keys are missing, the engine functions will return error messages instead of real answers.")
