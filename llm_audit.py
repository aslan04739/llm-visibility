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
except Exception as e:
    genai = None

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None

def get_default_prompts(brand="Votre Marque", competitors=["Concurrent A", "Concurrent B"]):
    comps_str = ", ".join(competitors)
    return {
        "Comparatif 1": f"Compare la visibilité et l'efficacité de {brand} vs {comps_str} sur leur marché principal.",
        "Comparatif 2": f"Pour les solutions métiers spécialisées, qui est le plus performant entre {brand} et {competitors[0]} ?",
        "Spécificité": f"Analyse la solution phare de {brand} face aux offres équivalentes de {comps_str}.",
        "Prix (ROI)": f"Comment est perçue la politique tarifaire de {brand} par rapport à {comps_str} ? Est-ce considéré comme un bon investissement ROI ?",
        "Fiabilité": f"Quelle est la perception des experts sur la durabilité et la fiabilité des produits {brand} face aux solutions de {comps_str} ?",
        "Forces (Brand)": f"Quels sont les points forts majeurs de {brand} selon les avis clients et experts du secteur ?",
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
        except Exception as e:
            client_openai = None
    if gemini_key and genai is not None:
        try:
            client_gemini = genai.Client(api_key=gemini_key)
        except Exception as e:
            client_gemini = None

def query_openai(prompt: str):
    if client_openai is None:
        return "Erreur OpenAI : clé API manquante ou client indisponible"
    try:
        resp = client_openai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.2, timeout=API_TIMEOUT_SECONDS)
        return resp.choices[0].message.content
    except Exception as e:
        return f"Erreur OpenAI: {e}"

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
        except Exception as e:
            time.sleep(2)
    return "Timeout Perplexity"

def analyze_response(text_response: str, brand: str = 'la marque', openai_for_eval_key: str = None):
    if client_openai is None:
        return 0.0, "Analyse impossible: client OpenAI non initialisé"
    if "Erreur" in text_response or "Timeout" in text_response:
        return 0.0, "Erreur de connexion API"
    eval_prompt = f"""
    Analyse cette réponse d'une IA concernant la marque {brand}.
    1. Donne une note globale de Perception entre -1.0 (Très Négatif) et +1.0 (Très Positif). 0 = Neutre.
    2. Rédige un résumé direct de l'information principale en EXACTEMENT 10 à 15 mots maximum, en français.
    Réponse brute : "{text_response[:1500]}"
    Format strict attendu : [Score numérique]|[Résumé court]
    """
    try:
        resp = client_openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": eval_prompt}], temperature=0, timeout=15)
        output = resp.choices[0].message.content.strip().split('|')
        return float(output[0]), output[1].strip()
    except Exception as e:
        return 0.0, f"Analyse impossible: {e}"

def run_granular_audit(prompts: dict, brand: str = 'la marque', openai_key: str = None, gemini_key: str = None, perplexity_key: str = None, progress_callback=None):
    """Run the audit: returns a pandas DataFrame with results.
    prompts: dict of {name: prompt}
    """
    setup_clients(openai_key=openai_key, gemini_key=gemini_key)
    engines = {"ChatGPT (GPT-4o)": query_openai, "Gemini (Flash)": query_gemini, "Perplexity (RAG)": lambda p: query_perplexity(p, perplexity_key)}
    data = []
    total = len(prompts) * len(engines)
    current = 0
    for theme, prompt in prompts.items():
        for engine_name, func in engines.items():
            current += 1
            if progress_callback:
                progress_callback(current, total, f"({current}/{total}) Interrogation de {engine_name} sur '{theme}'...")
            print(f"[{current}/{total}] Interrogation {engine_name} sur {theme} ...")
            
            raw_answer = func(prompt)
            
            if progress_callback:
                progress_callback(current, total, f"({current}/{total}) Évaluation de la réponse de {engine_name}...")
            print(f"[{current}/{total}] Évaluation de la réponse...")
            
            score, verbatim = analyze_response(raw_answer, brand=brand)
            data.append({
                "Axe Analysé": theme,
                "Moteur IA": engine_name,
                "Prompt Exact": prompt,
                "Score Perception": score,
                "Synthèse": verbatim
            })
            time.sleep(1)
    return pd.DataFrame(data)

def generate_visuals_return_figs(df: pd.DataFrame, target_brand: str = 'la marque'):
    """Generate matplotlib figures and return them in a dict.
    Does not save files by default.
    """
    # Wrap texts to avoid overlapping
    df['Axe Analysé'] = df['Axe Analysé'].apply(lambda x: textwrap.fill(str(x), width=30))
    sns.set_theme(style="whitegrid", context="talk")
    
    # CHARTE GRAPHIQUE
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'sans-serif'],
        'text.color': '#202124',
        'axes.labelcolor': '#202124',
        'xtick.color': '#202124',
        'ytick.color': '#202124',
        'axes.edgecolor': '#202124',
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF'
    })
    custom_palette = ["#47BDEF", "#202124", "#9AA0A6"]
    
    figs = {}

    # 1. Barplot
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    sns.barplot(data=df, y="Axe Analysé", x="Score Perception", hue="Moteur IA", palette=custom_palette, ax=ax1)
    ax1.set_title(f"Perception & Compétitivité de {target_brand} par Thématique", fontweight='bold', pad=20)
    ax1.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax1.set_xlabel("Score (Négatif = Faiblesse | Positif = Force)")
    fig1.tight_layout()
    figs['barplot'] = fig1

    # 2. Stripplot (consensus cloud)
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    sns.stripplot(data=df, x="Score Perception", y="Axe Analysé", hue="Moteur IA", size=8, jitter=True, palette=custom_palette, alpha=0.85, ax=ax2)
    ax2.set_title("Indice de Consensus des IA sur la marque et le marché", fontweight='bold', pad=20)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig2.tight_layout()
    figs['consensus'] = fig2

    # 3. Radar Multi-IA
    df_pivot = df.pivot_table(index='Axe Analysé', columns='Moteur IA', values='Score Perception').fillna(0)
    categories = [cat.split(" - ", 1)[1] if " - " in cat else cat for cat in df_pivot.index]
    categories = [textwrap.fill(c, 15) for c in categories]
    N = len(categories)
    
    if N == 0:
        return figs

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig3, ax3 = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax3.set_theta_offset(pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, size=10, fontweight='bold', color='#202124')
    ax3.set_rlabel_position(0)
    ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3.set_yticklabels(["Alerte", "Faible", "Neutre", "Bon", "Excellent"]) 
    ax3.set_ylim(-1, 1)

    engines = df['Moteur IA'].unique()
    for idx, engine in enumerate(engines):
        color = custom_palette[idx % len(custom_palette)]
        values = df_pivot[engine].tolist()
        values += values[:1]
        ax3.plot(angles, values, linewidth=2, linestyle='solid', label=engine, color=color)
        ax3.fill(angles, values, color=color, alpha=0.15)
        
    ax3.plot(angles, [0]*(N+1), color='#202124', linewidth=1, linestyle='--')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    ax3.set_title(f"Radar 360 : Benchmark & Empreinte de {target_brand}", size=14, fontweight='bold', pad=30)
    fig3.tight_layout()
    figs['radar'] = fig3

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
    ax3.plot(angles, values, linewidth=2, linestyle='solid', color='#47BDEF')
    ax3.fill(angles, values, color='#47BDEF', alpha=0.25)
    ax3.plot(angles, [0]*(N+1), color='#202124', linewidth=1, linestyle='--')
    ax3.set_title(f"Radar 360 : Benchmark & ADN {target_brand}", size=14, fontweight='bold', pad=20)
    fig3.tight_layout()
    figs['radar'] = fig3

    return figs
