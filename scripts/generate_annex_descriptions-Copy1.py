"""
COREP Annex Component Description Generator

Generiert KI-basierte Beschreibungen für COREP Annex ComponentLabels basierend auf:
- Text ohne Überschrift (aus ANNEX II INSTRUCTIONS)
- Excerpts (aus CRR)

Verwendet Mistral Chatbot API mit RAG Knowledge Base zur Vermeidung von Halluzinationen.
"""

import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
from fastprogress.fastprogress import progress_bar
import os

# API Configuration (übernommen aus chatbot.py)
CONFIG = {
    'web_ui_token': 'sk-15b54c10119c45f7a45e790a109d7c8b',
    'model_name': 'chatbot-mistral',
    'web_ui_base_url': 'https://chatbot-open-webui.apps.prod.w.oenb.co.at/',
    'knowledge_id': 'aace4dfd-3f4f-46da-9936-b38dc133e3e9',  # ITS AI USE CASE (COREP)
    
    # Anti-Hallucination LLM parameters
    'llm_parameters': {
        "temperature": 0.0,
        "max_tokens": 150,  # 2-4 Sätze
        "top_p": 0.5,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.3
    },
    
    'timeout': 60,
    'batch_size': 10,
    'output_dir': 'annex_descriptions_output'
}

# Prompt Template für ComponentLabel Beschreibungen
ANNEX_PROMPT_TEMPLATE = """You are a precise regulatory reporting assistant specialized in COREP Annex II and CRR regulations.

Your task: Generate a concise description (2-4 sentences) for the ComponentLabel "{component_label}" based ONLY on the provided context.

STRICT RULES:
1. Use ONLY information from the context below - do NOT add general knowledge or assumptions
2. Write 2-4 factual sentences explaining what this component means
3. If the context contains regulatory references (Articles, CRR), you may mention them naturally
4. Stay focused on the definition and purpose of this component
5. Do NOT speculate if information is insufficient - state what IS known from the context
6. Keep the response under 100 words

Table: {table}
Component Type: {component_type}
ComponentLabel: {component_label}

CONTEXT FROM ANNEX II INSTRUCTIONS:
{annex_text}

CONTEXT FROM CRR:
{crr_excerpts}

Generate a 2-4 sentence description based on the context above:"""


def call_chatbot_api(prompt, config):
    """
    Ruft Mistral Chatbot API auf (übernommen aus chatbot.py)
    """
    url = f"{config['web_ui_base_url']}api/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['web_ui_token']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config['model_name'],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    # Add LLM parameters
    if config.get('llm_parameters'):
        payload.update(config['llm_parameters'])
    
    # Add knowledge base
    if config.get('knowledge_id'):
        payload["files"] = [{'type': 'collection', 'id': config['knowledge_id']}]
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=config['timeout'])
        
        if response.status_code != 200:
            return f"API Error: HTTP {response.status_code}"
        
        json_response = response.json()
        
        if "choices" in json_response and json_response["choices"]:
            return json_response["choices"][0]["message"]["content"]
        else:
            return "API Error: Unexpected response format"
        
    except Exception as e:
        return f"Error: {str(e)}"


def load_and_prepare_data(columns_file, rows_file):
    """
    Lädt beide CSV-Dateien und bereitet unique Kombinationen vor
    """
    print("📂 Lade CSV-Dateien...")
    
    # CSV-Dateien laden
    df_columns = pd.read_csv(columns_file, encoding='utf-8')
    df_rows = pd.read_csv(rows_file, encoding='utf-8')
    
    print(f"✓ Columns: {len(df_columns)} Zeilen")
    print(f"✓ Rows: {len(df_rows)} Zeilen")
    
    # Füge Component Type hinzu
    df_columns['ComponentType'] = 'Column'
    df_rows['ComponentType'] = 'Row'
    
    # Kombiniere beide DataFrames
    df_combined = pd.concat([df_columns, df_rows], ignore_index=True)
    
    print(f"\n📊 Kombiniert: {len(df_combined)} Gesamtzeilen")
    
    # Gruppiere nach unique Table + ComponentType + ComponentLabel
    print("\n🔍 Gruppiere nach unique Kombinationen...")
    
    grouped = df_combined.groupby(['Table', 'ComponentType', 'ComponentLabel']).agg({
        'Text ohne Überschrift': lambda x: ' | '.join([str(text) for text in x if pd.notna(text) and str(text).strip()]),
        'Excerpts': lambda x: ' | '.join([str(excerpt) for excerpt in x if pd.notna(excerpt) and str(excerpt).strip()]),
        'Table_Short': 'first',
        'Column': 'first',  # Für Spalten
        'Row': 'first'      # Für Zeilen
    }).reset_index()
    
    # Bereinige leere Konkatenierungen
    grouped['Text ohne Überschrift'] = grouped['Text ohne Überschrift'].replace('', 'No instruction text available')
    grouped['Excerpts'] = grouped['Excerpts'].replace('', 'No CRR excerpt available')
    
    print(f"✓ {len(grouped)} unique Kombinationen identifiziert")
    
    return grouped


def generate_descriptions(df_unique, config):
    """
    Generiert Beschreibungen für jede unique Kombination
    """
    print(f"\n🤖 Starte Beschreibungsgenerierung...")
    print(f"📊 Zu verarbeiten: {len(df_unique)} unique ComponentLabels")
    print(f"🔧 LLM Parameters: {config['llm_parameters']}")
    
    results = []
    start_time = datetime.now()
    
    # Process in batches
    total_items = len(df_unique)
    
    for batch_start in range(0, total_items, config['batch_size']):
        batch_end = min(batch_start + config['batch_size'], total_items)
        batch_df = df_unique.iloc[batch_start:batch_end]
        
        print(f"\n📦 Batch {batch_start+1}-{batch_end} ({len(batch_df)} items)")
        
        for idx, row in progress_bar(list(batch_df.iterrows())):
            # Erstelle Prompt
            prompt = ANNEX_PROMPT_TEMPLATE.format(
                component_label=row['ComponentLabel'],
                table=row['Table'],
                component_type=row['ComponentType'],
                annex_text=row['Text ohne Überschrift'],
                crr_excerpts=row['Excerpts']
            )
            
            # API Call
            response_start = datetime.now()
            description = call_chatbot_api(prompt, config)
            processing_time = (datetime.now() - response_start).total_seconds()
            
            # Sammle Ergebnisse
            results.append({
                'Table': row['Table'],
                'Table_Short': row['Table_Short'],
                'ComponentType': row['ComponentType'],
                'Row_Column_Code': row['Row'] if row['ComponentType'] == 'Row' else row['Column'],
                'ComponentLabel': row['ComponentLabel'],
                'Generated_Description': description,
                'Context_Annex_Text': row['Text ohne Überschrift'],
                'Context_CRR_Excerpts': row['Excerpts'],
                'Response_Length_Chars': len(description),
                'Processing_Time_Seconds': processing_time,
                'Timestamp': datetime.now().isoformat()
            })
    
    # Erstelle Results DataFrame
    results_df = pd.DataFrame(results)
    
    # Berechne Statistiken
    total_time = (datetime.now() - start_time).total_seconds()
    avg_time = total_time / len(results_df) if len(results_df) > 0 else 0
    
    print(f"\n✅ Generierung abgeschlossen!")
    print(f"⏱️  Gesamtzeit: {total_time:.1f}s")
    print(f"⚡ Durchschnitt pro Item: {avg_time:.2f}s")
    
    return results_df


def save_results(results_df, config, is_test_mode=False):
    """
    Speichert Ergebnisse als CSV
    
    Args:
        results_df: DataFrame mit den Ergebnissen
        config: Konfigurations-Dictionary
        is_test_mode: Boolean, ob Test-Modus aktiv war
    """
    # Erstelle Output-Verzeichnis
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Erstelle Dateinamen mit Timestamp und Test-Marker
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = "_TEST" if is_test_mode else ""
    filename = f"annex_descriptions{mode_suffix}_{timestamp}.csv"
    filepath = output_dir / filename
    
    # Speichere als CSV
    results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 Ergebnisse gespeichert:")
    print(f"   {filepath}")
    
    # Erstelle auch eine Summary-Datei
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
    
    print(f"   {summary_file}")
    
    return filepath


def run_annex_description_generation(columns_csv_path, rows_csv_path):
    """
    Hauptfunktion - führt den kompletten Prozess durch
    
    Args:
        columns_csv_path: Pfad zur COLUMNS CSV-Datei
        rows_csv_path: Pfad zur ROWS CSV-Datei
    """
    print("=" * 60)
    print("COREP ANNEX COMPONENT DESCRIPTION GENERATOR")
    print("=" * 60)
    
    # Schritt 1: Daten laden und vorbereiten
    df_unique = load_and_prepare_data(columns_csv_path, rows_csv_path)
    
    # Preview anzeigen
    print("\n📋 Preview der zu verarbeitenden Daten (erste 3):")
    print(df_unique[['Table_Short', 'ComponentType', 'ComponentLabel']].head(3))
    
    # 🆕 Test-Modus Option
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
    
    # Bestätigung einholen
    proceed = input(f"\n▶️  Möchten Sie {len(df_to_process)} Beschreibungen generieren? (y/n): ").lower()
    if proceed not in ['y', 'yes', 'j', 'ja']:
        print("❌ Abgebrochen")
        return None
    
    # Schritt 2: Beschreibungen generieren
    results_df = generate_descriptions(df_to_process, CONFIG)
    
    # Schritt 3: Ergebnisse speichern
    output_file = save_results(results_df, CONFIG, is_test_mode)
    
    print("\n" + "=" * 60)
    print("✅ FERTIG!")
    print("=" * 60)
    
    return output_file


# Für direkten Aufruf oder Import
if __name__ == "__main__":
    # Beispiel-Verwendung mit korrekten Pfaden
    columns_file = "files/corep_extract/output_files/corep_annex_COLUMNS_20251009_151423.csv"
    rows_file = "files/corep_extract/output_files/corep_annex_ROWS_20251009_151423.csv"
    
    run_annex_description_generation(columns_file, rows_file)