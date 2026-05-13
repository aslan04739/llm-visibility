# LLM Visibility — Audit 360

This repo contains a small tool to run an "audit 360" across multiple LLMs (OpenAI, Gemini, Perplexity) and generate three visualizations (barplot, consensus stripplot, radar).

Quick start
- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Provide your API keys via environment variables (recommended):

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export PERPLEXITY_API_KEY="pplx-..."
```

- Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Usage notes
- The Streamlit sidebar accepts API keys and a JSON mapping of prompts. The default prompts are in `GRANULAR_PROMPTS` (from `llm_audit.py`).
- The app will call the three engines and return a DataFrame plus three figures shown in the UI. You can download the CSV of results.

Pushing to GitHub
- There's a helper script at `scripts/push_to_github.sh` that will commit and push to `origin` if configured. It will not add any secrets — do NOT commit API keys.

Security
- DO NOT store API keys in files. Use environment variables or the Streamlit password inputs (they are stored only in session).
# LLM Visibility — Streamlit Audit

This repo contains a small Streamlit app that runs a multi-engine (OpenAI / Gemini / Perplexity) audit across a set of prompts and renders three executive charts.

Files added:
- `llm_audit.py` — audit + plotting helpers
- `streamlit_app.py` — Streamlit UI entrypoint
- `requirements.txt` — Python dependencies

Quick deploy options

1) Streamlit Community Cloud (fastest)

   - Create a new app on https://share.streamlit.io and link this GitHub repository.
   - Ensure `streamlit_app.py` is the main file and `requirements.txt` is present.

2) Docker (portable)

   Build and run locally:

   ```bash
   docker build -t llm-audit:latest .
   docker run -p 8501:8501 llm-audit:latest
   ```

3) Heroku (simple VPS)

   - Ensure `requirements.txt` and `Procfile` exist.
   - Push to Heroku Git remote and deploy.

Security notes

- Never commit real API keys. Provide them using the Streamlit UI (sidebar) or via environment variables in your hosting platform.
- The app will still show error messages when keys are missing.

Want me to create the GitHub repo and attempt a Streamlit Cloud deploy for you? If yes, tell me the GitHub repo name and whether you want the repo private or public.
