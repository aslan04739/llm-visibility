import time
import textwrap
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi

try:
    from google import genai
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Default prompts copied/adapted from the notebook (no keys embedded)
TARGET_BRAND = "Sept Tools"
COMPETITORS = ["Husqvarna", "Hilti", "Flex"]

GRANULAR_PROMPTS = {
    "Bench - Ponçage Sol": f"Compare la visibilité et l'efficacité de {TARGET_BRAND} vs Husqvarna et Hilti pour le ponçage de sol béton professionnel.",
    "Bench - Ponçage Mur": f"Pour le ponçage mural (décapage, enduits), qui est le plus performant entre {TARGET_BRAND} et Flex ?",
    "Bench - Ponçage Plafond": f"Analyse la solution de ponçage de plafond de {TARGET_BRAND} (Gazellomur) face aux ponceuses girafes de {', '.join(COMPETITORS)}.",
    "Bench - Prix (ROI)": f"Comment est perçue la politique tarifaire de {TARGET_BRAND} par rapport à {', '.join(COMPETITORS)} ? Est-ce considéré comme un bon investissement ROI ?",
    "Bench - Durabilité": f"Quelle est la perception des IA sur la durabilité des moteurs brushless de {TARGET_BRAND} face aux solutions de {', '.join(COMPETITORS)} ?",
    "Brand - Forces": f"Quels sont les points forts majeurs de {TARGET_BRAND} selon les retours d'experts du BTP ?",
}

API_TIMEOUT_SECONDS = 60

client_openai = None
client_gemini = None

def setup_clients(openai_key: str = None, gemini_key: str = None):
    """Initialize API clients. Keys may be None; callers should handle missing keys."""
    global client_openai, client_gemini
    client_openai = None
    client_gemini = None
    if openai_key and OpenAI is not None:
        try:
            client_openai = OpenAI(api_key=openai_key)
        except Exception:
            client_openai = None
    if gemini_key and genai is not None:
        try:
            client_gemini = genai.Client(api_key=gemini_key)
        except Exception:
            client_gemini = None

def query_openai(prompt: str):
    if client_openai is None:
        return "Erreur OpenAI : clé API manquante ou client indisponible"
    try:
        resp = client_openai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.2, timeout=API_TIMEOUT_SECONDS)
        return resp.choices[0].message.content
    except Exception as e:
        return f"Erreur OpenAI : {e}"

def query_gemini(prompt: str):
    if client_gemini is None:
        return "Erreur Gemini : clé API manquante ou client indisponible"
    try:
        return client_gemini.models.generate_content(model='gemini-2.5-flash', contents=prompt).text
    except Exception as e:
        return f"Erreur Gemini : {e}"

def query_perplexity(prompt: str, perplexity_key: str = None, retries: int = 3):
    if not perplexity_key:
        return "Erreur Perplexity : clé API manquante"
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {perplexity_key}", "Content-Type": "application/json"}
    payload = {"model": "sonar", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT_SECONDS)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            else:
                return f"Erreur Perplexity HTTP {response.status_code}"
        except Exception:
            time.sleep(2)
    return "Timeout Perplexity"

def analyze_response(text_response: str, openai_for_eval_key: str = None):
    if client_openai is None:
        return 0.0, "Analyse impossible: client OpenAI non initialisé"
    if "Erreur" in text_response or "Timeout" in text_response:
        return 0.0, "Erreur de connexion API"
    eval_prompt = f"""
    Analyse cette réponse d'une IA concernant la marque {TARGET_BRAND}.
    1. Donne une note globale de Perception entre -1.0 (Très Négatif) et +1.0 (Très Positif). 0 = Neutre.
    2. Rédige un résumé direct de l'information principale en EXACTEMENT 10 à 15 mots maximum, en français.
    Réponse brute : "{text_response[:1500]}"
    Format strict attendu : [Score numérique]|[Résumé court]
    """
    try:
        resp = client_openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": eval_prompt}], temperature=0, timeout=15)
        output = resp.choices[0].message.content.strip().split('|')
        return float(output[0]), output[1].strip()
    except Exception:
        return 0.0, "Analyse impossible"

def run_granular_audit(prompts: dict, openai_key: str = None, gemini_key: str = None, perplexity_key: str = None):
    """Run the audit: returns a pandas DataFrame with results.
    prompts: dict of {name: prompt}
    """
    setup_clients(openai_key=openai_key, gemini_key=gemini_key)
    engines = {"ChatGPT (GPT-4o)": query_openai, "Gemini (Flash)": query_gemini, "Perplexity (RAG)": lambda p: query_perplexity(p, perplexity_key)}
    data = []
    for theme, prompt in prompts.items():
        for engine_name, func in engines.items():
            raw_answer = func(prompt)
            score, verbatim = analyze_response(raw_answer)
            data.append({
                "Axe Analysé": theme,
                "Moteur IA": engine_name,
                "Prompt Exact": prompt,
                "Score Perception": score,
                "Synthèse": verbatim
            })
            time.sleep(1)
    return pd.DataFrame(data)

def generate_visuals_return_figs(df: pd.DataFrame, target_brand: str = TARGET_BRAND):
    """Generate matplotlib figures and return them in a dict.
    Does not save files by default.
    """
    sns.set_theme(style="whitegrid", context="talk")
    figs = {}

    # 1. Barplot
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    sns.barplot(data=df, y="Axe Analysé", x="Score Perception", hue="Moteur IA", palette="Set2", ax=ax1)
    ax1.set_title(f"Perception & Compétitivité de {target_brand} par Thématique", fontweight='bold', pad=20)
    ax1.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax1.set_xlabel("Score (Négatif = Faiblesse | Positif = Force)")
    fig1.tight_layout()
    figs['barplot'] = fig1

    # 2. Stripplot (consensus cloud)
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    sns.stripplot(data=df, x="Score Perception", y="Axe Analysé", hue="Moteur IA", size=8, jitter=True, palette="Set1", alpha=0.85, ax=ax2)
    ax2.set_title("Indice de Consensus des IA sur la marque et le marché", fontweight='bold', pad=20)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig2.tight_layout()
    figs['consensus'] = fig2

    # 3. Radar
    df_mean = df.groupby('Axe Analysé')['Score Perception'].mean().reset_index()
    categories = [cat.split(" - ", 1)[1] if " - " in cat else cat for cat in df_mean['Axe Analysé'].tolist()]
    N = len(categories)
    values = df_mean['Score Perception'].tolist()
    if N == 0:
        # return empty figs if no data
        return figs
    values += values[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    fig3, ax3 = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax3.set_theta_offset(pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=9, fontweight='bold')
    ax3.set_rlabel_position(0)
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticklabels(["Alerte", "Faible", "Neutre", "Bon", "Excellent"]) 
    ax3.set_ylim(-1, 1)
    ax3.plot(angles, values, linewidth=2, linestyle='solid', color='#2ecc71')
    ax3.fill(angles, values, color='#2ecc71', alpha=0.25)
    ax3.plot(angles, [0]*(N+1), color='black', linewidth=1, linestyle='--')
    ax3.set_title(f"Radar 360 : Benchmark & ADN {target_brand}", size=14, fontweight='bold', pad=20)
    fig3.tight_layout()
    figs['radar'] = fig3

    return figs
