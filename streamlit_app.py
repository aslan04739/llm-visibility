import streamlit as st
import json
import pandas as pd
from llm_audit import get_default_prompts, run_granular_audit, generate_visuals_return_figs

st.set_page_config(page_title="Audit de Visibilité LLM", layout="wide")

st.write("")
st.write("")

st.title("Audit de Visibilité LLM — Diagnostic 360°")
st.markdown("---")

st.markdown("""
### Évaluez et optimisez le positionnement de votre marque
Cet outil vous permet de mesurer comment les moteurs d'Intelligence Artificielle (ChatGPT, Gemini, Perplexity) perçoivent vos produits et vos offres. 
**Sécuriser votre visibilité LLM est un enjeu d’acquisition business majeur** : si une IA vous oublie ou vous cite comme peu fiable, vos clients finaux cibleront vos concurrents. 

Exemple de concept clé : le **Score de Perception**. 
*Un score de `-1.0` indique que l'IA déconseille votre produit (Alerte business, risque de perte de parts de marché).* 
*Un score de `+1.0` signifie que vous êtes cité comme la référence absolue (Opportunité stratégique).* 
""")
st.write("")

with st.sidebar.form(key='keys_and_prompts'):
    st.header("⚙️ Configuration du Diagnostic")
    st.markdown("Nous vous invitons à renseigner vos clés d'API professionnelles afin de lancer l'audit.")
    openai_key = st.text_input("Clé API OpenAI (Requis pour l'analyse)", type="password")
    gemini_key = st.text_input("Clé API Gemini (Optionnel)", type="password")
    perplexity_key = st.text_input("Clé API Perplexity (Optionnel)", type="password")

    st.markdown("---")
    st.markdown("**Marque cible & Concurrents**")
    brand_name = st.text_input("Marque à auditer", value="Votre Marque")
    competitors_str = st.text_input("Concurrents (séparés par des virgules)", value="Concurrent A, Concurrent B")

    st.write("")
    use_defaults = st.checkbox("Générer/utiliser les requêtes (prompts) par défaut", value=True)
    
    competitors_list = [c.strip() for c in competitors_str.split(',') if c.strip()]
    granular_prompts = get_default_prompts(brand=brand_name, competitors=competitors_list if competitors_list else ["Concurrents"])
    sample_json = json.dumps(granular_prompts, ensure_ascii=False, indent=2)
    
    st.markdown("**Personnalisez vos requêtes pour auditer un segment business précis :**")
    prompts_text = st.text_area("Vos stratégies de requêtes (format JSON)", value=sample_json if use_defaults else "{\n  \n}", height=240)

    submit = st.form_submit_button("Sauvegarder la configuration")

if submit:
    st.success("Vos paramètres ont bien été enregistrés. Vous pouvez lancer le diagnostic.")

st.write("")
run_btn = st.button("Lancer le diagnostic 360°", type="primary")

if run_btn:
    try:
        prompts = json.loads(prompts_text)
        if not isinstance(prompts, dict):
            st.error("Le format JSON des requêtes est invalide. Nous attendons un objet associant le nom de l'axe à la requête texte.")
            st.stop()
    except Exception as e:
        st.error(f"Le format JSON contient une erreur de syntaxe : {e}")
        st.stop()


    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def on_progress(current, total, msg):
        progress_bar.progress(current / total)
        status_text.write(msg)

    with st.spinner("L'audit est en cours... Cette opération prend environ 1 à 2 minutes (18 requêtes multi-IA)."):
        df = run_granular_audit(prompts, brand=brand_name, openai_key=openai_key or None, gemini_key=gemini_key or None, perplexity_key=perplexity_key or None, progress_callback=on_progress)
        
    progress_bar.empty()
    status_text.empty()


    st.success("Diagnostic complété avec succès ! Voici vos indicateurs de compétitivité.")

    figs = generate_visuals_return_figs(df, target_brand=brand_name)

    col1, col2 = st.columns(2)
    if 'barplot' in figs:
        with col1:
            st.subheader("Analyse de compétitivité par thématique")
            st.write("Ce graphique vous permet d'identifier vos atouts concurrentiels et les axes d'amélioration critiques pour votre positionnement.")
            st.pyplot(figs['barplot'])
            
    if 'consensus' in figs:
        with col2:
            st.subheader("Indice de Consensus des IA")
            st.write("Visualisez facilement les points d'accord ou de fragmentation entre les différents moteurs d'IA du marché.")
            st.pyplot(figs['consensus'])

    if 'radar' in figs:
        st.write("")
        st.subheader("Empreinte globale de la marque (Radar 360)")
        st.write("Aperçu synthétique de votre empreinte globale pour guider vos prochains choix stratégiques en matière de présence de marque.")
        st.pyplot(figs['radar'])

    st.markdown("---")
    st.subheader("Données brutes & Recommandations stratégiques")
    st.dataframe(df)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Télécharger vos données (CSV)", data=csv, file_name='audit_360_resultats.csv', mime='text/csv')

