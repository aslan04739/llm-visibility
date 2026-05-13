import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "audit_360_sept_tools.csv"


def load_data(path):
    return pd.read_csv(path)


def classify_row(ax):
    if pd.isna(ax):
        return None
    a = str(ax).lower()
    if "argument pour" in a or "arguments pour" in a or "argument pour" in a:
        return "Pour"
    if "argument contre" in a or "arguments contre" in a or "argument contre" in a:
        return "Contre"
    if "argument pour" in a.upper() or "argument contre" in a.upper():
        return "Autre"
    return None


def extract_arguments(df):
    # identify Pour / Contre rows using common keywords in 'Axe Analysé'
    mask_pour = df['Axe Analysé'].str.contains('Arguments Pour|Argument POUR|SWOT - Argument POUR', case=False, na=False)
    mask_contre = df['Axe Analysé'].str.contains('Arguments Contre|Argument CONTRE|SWOT - Argument CONTRE', case=False, na=False)
    df_args = df[mask_pour | mask_contre].copy()
    df_args['Type'] = df_args['Axe Analysé'].apply(lambda x: 'Pour' if ('pour' in str(x).lower()) else 'Contre')
    return df_args


def plot_scores(df_args, out_path="arguments_for_against.png"):
    # pivot table: index = Moteur IA, columns = Type, values = mean Score
    pivot = df_args.pivot_table(index='Moteur IA', columns='Type', values='Score Perception', aggfunc='mean')
    pivot = pivot.fillna(0)

    ax = pivot.plot(kind='bar', figsize=(8,5))
    ax.set_ylabel('Score Perception (moyenne)')
    ax.set_title('Arguments Pour vs Contre — moyenne par moteur IA')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved chart to {out_path}")


def write_summaries(df_args, df, out_args="arguments_summary.txt", out_gaps="brand_gaps.txt"):
    with open(out_args, 'w', encoding='utf-8') as f:
        for moteur, group in df_args.groupby('Moteur IA'):
            f.write(f"=== {moteur} ===\n")
            for _, row in group.iterrows():
                f.write(f"- Type: {row['Type']} | Prompt: {row.get('Prompt Exact','')}\n")
                f.write(f"  Synthèse: {row.get('Synthèse','')}\n")
            f.write('\n')
    print(f"Saved arguments summary to {out_args}")

    # Brand lacunes: prefer rows mentioning 'Faiblesses' or negative brand scores
    mask_brand_failles = df['Axe Analysé'].str.contains('Faiblesses|faiblesse', case=False, na=False)
    mask_brand = df['Axe Analysé'].str.contains('Brand|brand', case=False, na=False)
    brand_rows = df[mask_brand & (mask_brand_failles | (df['Score Perception'] < 0))].copy()
    with open(out_gaps, 'w', encoding='utf-8') as f:
        for moteur, group in brand_rows.groupby('Moteur IA'):
            f.write(f"=== {moteur} ===\n")
            for _, row in group.sort_values('Score Perception').iterrows():
                f.write(f"- Axe: {row['Axe Analysé']} | Score: {row['Score Perception']}\n")
                f.write(f"  Synthèse: {row.get('Synthèse','')}\n")
            f.write('\n')
    print(f"Saved brand gaps to {out_gaps}")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV file not found: {CSV_PATH}")
        return
    df = load_data(CSV_PATH)
    df_args = extract_arguments(df)
    if df_args.empty:
        print("No 'Arguments Pour/Contre' rows found in the CSV.")
        return
    plot_scores(df_args)
    write_summaries(df_args, df)


if __name__ == '__main__':
    main()
