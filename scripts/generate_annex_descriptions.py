"""
COREP Annex Component Description Generator (Prompt- & Engineering-optimiert)

Generiert KI-basierte Beschreibungen für COREP Annex ComponentLabels basierend auf:
- Text ohne Überschrift (aus ANNEX II INSTRUCTIONS)
- Excerpts (aus CRR)

Verbesserungen:
A) Strenger Prompt inkl. Fallback-Satz (ohne Code-Guards):
   -> If context is unclear, prefer: “The available information is insufficient for a precise definition.”
   -> Keine externen Annahmen; CRR-Artikel nur nennen, wenn sie im Excerpt stehen
   -> Saubere Kontext-Abgrenzung im Prompt

C) Engineering:
   - Deterministische LLM-Parameter
   - Token aus ENV (WEB_UI_TOKEN), nicht hardcodiert
   - API-Smoketest
   - Kontext-Trunkierung, sauberes Error-Handling, Längenbegrenzung
   - Robuste Ausgabe (CSV + Summary)

Verwendet Mistral Chatbot API mit RAG-Knowledge-Base.
"""

import os
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from fastprogress.fastprogress import progress_bar

# ---------------------------
# Konfiguration
# ---------------------------

CONFIG = {
    'web_ui_token': os.getenv('WEB_UI_TOKEN'),  # <-- ENV statt Hardcode!
    'model_name': 'chatbot-mistral',
    'web_ui_base_url': 'https://chatbot-open-webui.apps.prod.w.oenb.co.at/',
    'knowledge_id': 'aace4dfd-3f4f-46da-9936-b38dc133e3e9',  # ITS AI USE CASE (COREP)

    # Deterministische LLM-Parameter
    'llm_parameters': {
        "temperature": 0.0,
        "max_tokens": 150,   # 2–4 Sätze
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    },

    'timeout': 60,
    'batch_size': 10,
    'output_dir': 'annex_descriptions_output',

    # Kontext-Limits (Zeichen)
    'annex_maxlen': 6000,
    'crr_maxlen': 6000
}

# ---------------------------
# Prompt-Vorlage (A)
# ---------------------------

ANNEX_PROMPT_TEMPLATE = """You are a precise regulatory reporting assistant specialized in COREP Annex II and the CRR.

Your task: Generate a concise description (2–4 sentences) for the ComponentLabel "{component_label}" based ONLY on the provided context.

STRICT RULES:
1) Use ONLY information from the CONTEXT below. Ignore any external knowledge or assumptions.
2) If a term’s definition (e.g., “financial sector entity”, “significant investment”) is NOT given in the context, do NOT define it.
3) Cite CRR articles ONLY if their numbers appear verbatim in the CRR excerpts. If none appear, refer generically to “the CRR” without article numbers.
4) Keep 2–4 factual sentences, ≤100 words, neutral tone. No examples.
5) If the context conflicts, say what is consistent and omit the rest.
6) If the context is unclear or insufficient, prefer: “The available information is insufficient for a precise definition.”

Table: {table}
Component Type: {component_type}
ComponentLabel: {component_label}

CONTEXT FROM ANNEX II INSTRUCTIONS:
<<<ANNEX>>>
{annex_text}
<<<END ANNEX>>>

CONTEXT FROM CRR:
<<<CRR>>>
{crr_excerpts}
<<<END CRR>>>

Generate a 2–4 sentence description that obeys the STRICT RULES above."""

# ---------------------------
# Utilities (C)
# ---------------------------

def truncate(s: str, maxlen: int) -> str:
    """Trunkiert lange Strings am Wortende."""
    if not isinstance(s, str):
        return ""
    if len(s) <= maxlen:
        return s
    cut = s[:maxlen]
    return cut.rsplit(' ', 1)[0] + ' …'

def enforce_length(text: str, max_words: int = 100) -> str:
    """Begrenzt hart auf max_words."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    return " ".join(words[:max_words])

# ---------------------------
# API (C)
# ---------------------------

def call_chatbot_api(prompt: str, config: dict):
    """
    Ruft Mistral Chatbot API auf.
    Rückgabe: (content_str_oder_None, error_message_oder_None)
    """
    url = f"{config['web_ui_base_url']}api/chat/completions"
    token = config.get('web_ui_token')
    if not token:
        return None, "Missing WEB_UI_TOKEN environment variable."

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config['model_name'],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    # LLM-Parameter
    if config.get('llm_parameters'):
        payload.update(config['llm_parameters'])

    # Knowledge-Base
    if config.get('knowledge_id'):
        payload["files"] = [{'type': 'collection', 'id': config['knowledge_id']}]

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=config['timeout'])
        if resp.status_code != 200:
            snippet = resp.text[:300].replace("\n", " ")
            return None, f"API Error: HTTP {resp.status_code} – {snippet}"
        js = resp.json()
        if "choices" in js and js["choices"]:
            content = js["choices"][0]["message"]["content"]
            return content, None
        return None, f"API Error: Unexpected response format: {list(js.keys())}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def test_api_connection(config: dict) -> None:
    """Kleiner Smoke-Test für Token/Endpoint (C)."""
    prompt = "Reply with exactly: OK"
    content, err = call_chatbot_api(prompt, config)
    if err:
        print(f"❌ API connectivity problem: {err}")
    else:
        print(f"✅ API response sample: {content[:60]!r}")

# ---------------------------
# Datenvorbereitung
# ---------------------------

def load_and_prepare_data(columns_file: str, rows_file: str) -> pd.DataFrame:
    """
    Lädt beide CSV-Dateien und bereitet unique Kombinationen vor.
    """
    print("📂 Lade CSV-Dateien...")

    df_columns = pd.read_csv(columns_file, encoding='utf-8')
    df_rows = pd.read_csv(rows_file, encoding='utf-8')

    print(f"✓ Columns: {len(df_columns)} Zeilen")
    print(f"✓ Rows: {len(df_rows)} Zeilen")

    df_columns['ComponentType'] = 'Column'
    df_rows['ComponentType'] = 'Row'

    df_combined = pd.concat([df_columns, df_rows], ignore_index=True)
    print(f"\n📊 Kombiniert: {len(df_combined)} Gesamtzeilen")

    grouped = df_combined.groupby(['Table', 'ComponentType', 'ComponentLabel']).agg({
        'Text ohne Überschrift': lambda x: ' | '.join([str(t) for t in x if pd.notna(t) and str(t).strip()]),
        'Excerpts': lambda x: ' | '.join([str(e) for e in x if pd.notna(e) and str(e).strip()]),
        'Table_Short': 'first',
        'Column': 'first',  # für Spalten
        'Row': 'first'      # für Zeilen
    }).reset_index()

    # Platzhalter für leere Kontexte
    grouped['Text ohne Überschrift'] = grouped['Text ohne Überschrift'].replace('', 'No instruction text available')
    grouped['Excerpts'] = grouped['Excerpts'].replace('', 'No CRR excerpt available')

    print(f"✓ {len(grouped)} unique Kombinationen identifiziert")
    return grouped

# ---------------------------
# Generation
# ---------------------------

def generate_descriptions(df_unique: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generiert Beschreibungen für jede unique Kombination.
    (ohne Code-Guards – Prompt regelt zurückhaltung, Fallback nur bei API-Fehlern)
    """
    print(f"\n🤖 Starte Beschreibungsgenerierung...")
    print(f"📊 Zu verarbeiten: {len(df_unique)} unique ComponentLabels")
    print(f"🔧 LLM Parameters: {config['llm_parameters']}")

    results = []
    start_time = datetime.now()
    total_items = len(df_unique)

    for batch_start in range(0, total_items, config['batch_size']):
        batch_end = min(batch_start + config['batch_size'], total_items)
        batch_df = df_unique.iloc[batch_start:batch_end]

        print(f"\n📦 Batch {batch_start+1}-{batch_end} ({len(batch_df)} items)")

        for idx, row in progress_bar(list(batch_df.iterrows())):
            # Kontext trunkiert (C)
            annex_text = truncate(row['Text ohne Überschrift'], config['annex_maxlen'])
            crr_text = truncate(row['Excerpts'], config['crr_maxlen'])

            # Prompt (A)
            prompt = ANNEX_PROMPT_TEMPLATE.format(
                component_label=row['ComponentLabel'],
                table=row['Table'],
                component_type=row['ComponentType'],
                annex_text=annex_text,
                crr_excerpts=crr_text
            )

            # API-Call (C)
            response_start = datetime.now()
            description, err = call_chatbot_api(prompt, config)
            processing_time = (datetime.now() - response_start).total_seconds()

            api_error = None
            if err or not description:
                api_error = err or "Empty response"
                description = "The available information is insufficient for a precise definition."
            else:
                description = enforce_length(description, max_words=100)

            # Ergebnis sammeln
            results.append({
                'Table': row['Table'],
                'Table_Short': row['Table_Short'],
                'ComponentType': row['ComponentType'],
                'Row_Column_Code': row['Row'] if row['ComponentType'] == 'Row' else row['Column'],
                'ComponentLabel': row['ComponentLabel'],
                'Generated_Description': description,
                'Context_Annex_Text': annex_text,
                'Context_CRR_Excerpts': crr_text,
                'Response_Length_Chars': len(description or ""),
                'Processing_Time_Seconds': processing_time,
                'Api_Error': api_error,
                'Timestamp': datetime.now().isoformat()
            })

    results_df = pd.DataFrame(results)

    total_time = (datetime.now() - start_time).total_seconds()
    avg_time = total_time / len(results_df) if len(results_df) > 0 else 0

    print(f"\n✅ Generierung abgeschlossen!")
    print(f"⏱️  Gesamtzeit: {total_time:.1f}s")
    print(f"⚡ Durchschnitt pro Item: {avg_time:.2f}s")

    return results_df

# ---------------------------
# Speichern
# ---------------------------

def save_results(results_df: pd.DataFrame, config: dict, is_test_mode: bool = False) -> Path:
    """
    Speichert Ergebnisse als CSV + Summary.
    """
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = "_TEST" if is_test_mode else ""
    filename = f"annex_descriptions{mode_suffix}_{timestamp}.csv"
    filepath = output_dir / filename

    results_df.to_csv(filepath, index=False, encoding='utf-8-sig')

    print(f"\n💾 Ergebnisse gespeichert:")
    print(f"   {filepath}")

    # Summary-Datei
    summary_file = output_dir / f"summary{mode_suffix}_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("COREP ANNEX DESCRIPTION GENERATION - SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Mode: {'🧪 TEST MODE (First 20 items)' if is_test_mode else '✅ FULL RUN'}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total Items: {len(results_df)}\n")
        f.write(f"Total Tables: {results_df['Table_Short'].nunique()}\n")
        f.write(f"Rows: {len(results_df[results_df['ComponentType'] == 'Row'])}\n")
        f.write(f"Columns: {len(results_df[results_df['ComponentType'] == 'Column'])}\n\n")
        f.write(f"LLM Parameters:\n")
        for key, value in config['llm_parameters'].items():
            f.write(f"  - {key}: {value}\n")
        f.write(f"\nAverage Response Length: {results_df['Response_Length_Chars'].mean():.0f} chars\n")
        f.write(f"Average Processing Time: {results_df['Processing_Time_Seconds'].mean():.2f}s\n")
        f.write(f"API Errors: {results_df['Api_Error'].notna().sum()} items\n")

    print(f"   {summary_file}")
    return filepath

# ---------------------------
# Main-Orchestrierung
# ---------------------------

def run_annex_description_generation(columns_csv_path: str, rows_csv_path: str):
    """
    Hauptfunktion - führt den kompletten Prozess durch.
    """
    print("=" * 60)
    print("COREP ANNEX COMPONENT DESCRIPTION GENERATOR (Prompt & Eng Tweaks)")
    print("=" * 60)

    # API-Smoke-Test (C)
    print("🔎 API smoke test …")
    test_api_connection(CONFIG)

    # Schritt 1: Daten laden und vorbereiten
    df_unique = load_and_prepare_data(columns_csv_path, rows_csv_path)

    # Preview
    print("\n📋 Preview der zu verarbeitenden Daten (erste 3):")
    print(df_unique[['Table_Short', 'ComponentType', 'ComponentLabel']].head(3))

    # Test-Modus
    print(f"\n📊 Insgesamt: {len(df_unique)} unique ComponentLabels gefunden")
    test_mode = input("🧪 Test-Modus: Nur erste 20 Einträge verarbeiten? (y/n): ").lower()
    is_test_mode = False
    if test_mode in ['y', 'yes', 'j', 'ja']:
        df_to_process = df_unique.head(20).copy()
        is_test_mode = True
        print(f"✅ Test-Modus aktiviert: Verarbeite {len(df_to_process)} Einträge")
    else:
        df_to_process = df_unique
        print(f"✅ Vollständiger Modus: Verarbeite {len(df_to_process)} Einträge")

    # Bestätigung
    proceed = input(f"\n▶️  Möchten Sie {len(df_to_process)} Beschreibungen generieren? (y/n): ").lower()
    if proceed not in ['y', 'yes', 'j', 'ja']:
        print("❌ Abgebrochen")
        return None

    # Schritt 2: Generierung
    results_df = generate_descriptions(df_to_process, CONFIG)

    # Schritt 3: Speichern
    output_file = save_results(results_df, CONFIG, is_test_mode)

    print("\n" + "=" * 60)
    print("✅ FERTIG!")
    print("=" * 60)

    return output_file

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    # Beispiel-Pfade anpassen
    columns_file = "files/corep_extract/output_files/corep_annex_COLUMNS_20251009_151423.csv"
    rows_file = "files/corep_extract/output_files/corep_annex_ROWS_20251009_151423.csv"

    run_annex_description_generation(columns_file, rows_file)
