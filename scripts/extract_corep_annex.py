"""
extract_corep_annex.py - Implementierung der COREP Annex 2 Extraktion
"""

import docx
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import pickle

# Absolute Imports
import scripts.utils_annex as utils_annex
import scripts.utils_crr as utils_crr
import scripts.utils_match_chatbot as utils_match


def expand_column_ranges(df, column_name='Column'):
    """Expandiert Ranges und Listen in der Column-Spalte zu einzelnen Zeilen."""
    expanded_rows = []
    
    for idx, row in df.iterrows():
        column_value = str(row[column_name]).strip()
        column_values = []
        
        # Pattern 1: Ranges mit "and"
        if ' and ' in column_value.lower():
            parts = re.split(r'\s+and\s+', column_value, flags=re.IGNORECASE)
            for part in parts:
                column_values.extend(_parse_column_part(part.strip()))
        
        # Pattern 2: Komma-getrennte Liste
        elif ',' in column_value:
            parts = [p.strip() for p in column_value.split(',')]
            for part in parts:
                column_values.extend(_parse_column_part(part))
        
        # Pattern 3: Einfacher Range mit "-" oder "to"
        elif '-' in column_value or ' to ' in column_value.lower():
            column_values.extend(_parse_column_part(column_value))
        
        # Pattern 4: Einzelner Wert
        else:
            column_values.append(column_value)
        
        # Für jeden Wert eine neue Zeile
        for col_val in column_values:
            new_row = row.copy()
            new_row[column_name] = col_val
            expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def _parse_column_part(part):
    """Parst einen einzelnen Teil (Range oder Einzelwert)."""
    part = part.strip()
    
    # Range-Pattern mit "-" oder "to"
    pattern_dash = r'^(\d+)\s*-\s*(\d+)$'
    pattern_to = r'^(\d+)\s+to\s+(\d+)$'
    
    range_match = re.match(pattern_dash, part) or re.match(pattern_to, part, re.IGNORECASE)
    
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        padding = len(range_match.group(1))
        return [str(i).zfill(padding) for i in range(start, end + 1)]
    else:
        return [part]


def get_component_label(table_short, code, all_trees, component_type):
    """
    Holt das ComponentLabel aus der Baumstruktur.
    
    Args:
        table_short: Kurzer Table-Code (z.B. "C 01.00")
        code: Row/Column Code (z.B. "0015")
        all_trees: Dictionary mit allen Baumstrukturen
        component_type: "Table row" oder "Table column"
        
    Returns:
        str: ComponentLabel oder "N/A" wenn nicht gefunden
    """
    # Suche passenden Baum
    for (tree_table_code, tree_comp_type), roots in all_trees.items():
        # Extrahiere kurzen Code aus Baum-Key
        tree_short = extract_table_code_prefix(tree_table_code)
        
        # Matching: Table_Short und ComponentType müssen passen
        if tree_short == table_short and tree_comp_type == component_type:
            # Suche Knoten mit componentcode
            label = find_label_in_tree(roots, code)
            if label:
                return label
    
    return "N/A"


def find_label_in_tree(roots, componentcode):
    """
    Sucht rekursiv nach einem Knoten mit dem gegebenen componentcode.
    
    Args:
        roots: Liste von Wurzelknoten
        componentcode: Zu suchender Code (z.B. "0015")
        
    Returns:
        str: componentlabel oder None
    """
    if not roots:
        return None
    
    for root in roots:
        result = find_label_recursive(root, componentcode)
        if result:
            return result
    
    return None


def find_label_recursive(node, componentcode):
    """
    Rekursive Suche nach componentcode im Baum.
    
    Args:
        node: Aktueller Knoten
        componentcode: Gesuchter Code
        
    Returns:
        str: componentlabel oder None
    """
    # Vergleiche Codes (mit und ohne Zero-Padding)
    node_code = str(node.componentcode).strip()
    search_code = str(componentcode).strip()
    
    # Beide auf gleiche Länge bringen (4 Zeichen mit Zeros)
    node_code_padded = node_code.zfill(4)
    search_code_padded = search_code.zfill(4)
    
    if node_code_padded == search_code_padded:
        return node.componentlabel
    
    # Suche in Kindern
    if hasattr(node, 'children'):
        for child in node.children:
            result = find_label_recursive(child, componentcode)
            if result:
                return result
    
    return None


def extract_table_code_prefix(full_table_name):
    """
    Extrahiert den Tabellen-Code vor dem ersten Trennzeichen.
    
    Beispiele:
    - "C 01.00 - OWN FUNDS" -> "C 01.00"
    - "C 06.01 – GROUP SOLVENCY" -> "C 06.01"
    - "C 10.01 and C 10.02" -> "C 10.01"
    - "C 17.01: Operational" -> "C 17.01"
    """
    if pd.isna(full_table_name) or str(full_table_name) == "nan":
        return ""
    
    full_name = str(full_table_name).strip()
    
    # Pattern: "C XX.XX" oder "C XX.XX.X"
    pattern = r'^(C\s+\d+\.\d+(?:\.\d+)?)'
    match = re.match(pattern, full_name)
    
    if match:
        return match.group(1).strip()
    else:
        # Fallback: Trennzeichen
        for separator in [' –', ' -', ':', ' and']:
            if separator in full_name:
                return full_name.split(separator)[0].strip()
        return full_name


def extract_corep_annex():
    """Hauptfunktion für die COREP Annex Extraktion"""
    
    print("="*60)
    print("COREP ANNEX 2 EXTRACTION - START")
    print("="*60)
    
    # Pfade
    base_path = Path("files/corep_extract")
    annex_path = base_path / "Annex 2 (Solvency).docx"
    crr_path = base_path / "CRR_CELEX_02013R0575-20250101_EN_TXT.pdf"
    output_path = base_path / "output_files"
    tree_path = Path("tree_structures/baumstruktur_COREP_3_2.pkl")
    
    if not annex_path.exists():
        print(f"❌ Fehler: {annex_path} nicht gefunden!")
        return None, None
    if not crr_path.exists():
        print(f"❌ Fehler: {crr_path} nicht gefunden!")
        return None, None
    if not tree_path.exists():
        print(f"❌ Fehler: {tree_path} nicht gefunden!")
        return None, None
    
    print(f"✓ Dateien gefunden:")
    print(f"  - Annex: {annex_path}")
    print(f"  - CRR: {crr_path}")
    print(f"  - Trees: {tree_path}")
    
    # STEP 1: ANNEX 2 WORD-DOKUMENT
    print("\n" + "="*60)
    print("STEP 1: Annex 2 (Solvency) einlesen")
    print("="*60)
    
    print("📄 Lade Word-Dokument...")
    doc = docx.Document(annex_path)
    
    print("🔍 Extrahiere Tabellen...")
    content_dict = utils_annex.get_content_dict(doc)
    print(f"✓ {len(content_dict)} Tabellen gefunden")
    
    print("\n🔧 Extrahiere Metadaten (ROWS & COLUMNS)...")
    annex_rows, annex_columns = utils_annex.extract_metadata(content_dict)
    print(f"✓ ROWS: {len(annex_rows)} Einträge")
    print(f"✓ COLUMNS: {len(annex_columns)} Einträge")
    
    # STEP 2: CRR PDF
    print("\n" + "="*60)
    print("STEP 2: CRR PDF einlesen")
    print("="*60)
    
    print("📄 Lade CRR PDF...")
    start_page = 3
    end_page = 847
    
    HEADER_INDENT = utils_crr.get_indents(str(crr_path), ref_page=2)
    print(f"✓ Header Indent: {HEADER_INDENT}")
    
    print(f"\n🔧 Extrahiere CRR-Artikel (Seiten {start_page}-{end_page})...")
    print("   ⏳ Dies kann einige Minuten dauern...")
    
    full_doc = utils_crr.Extract(
        str(crr_path), 
        start_page=start_page, 
        end_page=end_page, 
        header_text_indent=HEADER_INDENT
    )
    row_list = full_doc.crr_extraction()
    
    print(f"✓ {len(row_list)} CRR-Einträge extrahiert")
    
    crr_refs = pd.DataFrame(
        row_list, 
        columns=["article", "paragraph", "number", "point", "subpoint", "text"]
    )
    crr_refs = crr_refs[crr_refs["text"] != ""]
    crr_refs.set_index(["article", "paragraph", "number", "point", "subpoint"], inplace=True)
    
    print(f"✓ CRR DataFrame bereinigt: {len(crr_refs)} Einträge")
    
    # STEP 3: LEGAL REFERENCES
    print("\n" + "="*60)
    print("STEP 3: Legal References aus CRR extrahieren")
    print("="*60)
    
    print("🔗 Extrahiere CRR-Texte für ROWS...")
    full_text_rows = utils_match.get_text(annex_rows, crr_refs)
    
    print("🔗 Extrahiere CRR-Texte für COLUMNS...")
    full_text_columns = utils_match.get_text(annex_columns, crr_refs)
    
    excerpts_df_rows = annex_rows.copy()
    excerpts_df_columns = annex_columns.copy()
    
    excerpts_df_rows["Excerpts"] = full_text_rows
    excerpts_df_columns["Excerpts"] = full_text_columns
    
    print(f"✓ Excerpts hinzugefügt")
    
    # STEP 4: ROW UND COLUMN RANGES EXPANDIEREN
    print("\n" + "="*60)
    print("STEP 4: Row und Column Ranges expandieren")
    print("="*60)
    
    # 4a: ROWS expandieren
    print("🔧 Prüfe ROWS auf Ranges/Listen...")
    rows_before = len(excerpts_df_rows)
    
    rows_with_ranges = excerpts_df_rows[
        excerpts_df_rows['Row'].str.contains(r'[-,]|and|to', case=False, na=False, regex=True)
    ]
    
    if len(rows_with_ranges) > 0:
        print(f"✓ {len(rows_with_ranges)} ROWS mit Ranges/Listen gefunden")
        print("\n📊 Beispiele vor Expansion:")
        for idx, row in rows_with_ranges.head(3).iterrows():
            print(f"  • {row['Table'][:40]}... - Row: '{row['Row']}'")
        
        print("\n🔄 Expandiere Ranges/Listen für ROWS...")
        excerpts_df_rows = expand_column_ranges(excerpts_df_rows, column_name='Row')
        
        rows_after = len(excerpts_df_rows)
        print(f"✓ ROWS Expansion: {rows_before} → {rows_after} Zeilen (+{rows_after - rows_before})")
    else:
        print("✓ Keine Ranges/Listen in ROWS gefunden")
    
    # 4b: COLUMNS expandieren
    print("\n🔧 Prüfe COLUMNS auf Ranges/Listen...")
    cols_before = len(excerpts_df_columns)
    
    columns_with_ranges = excerpts_df_columns[
        excerpts_df_columns['Column'].str.contains(r'[-,]|and|to', case=False, na=False, regex=True)
    ]
    
    if len(columns_with_ranges) > 0:
        print(f"✓ {len(columns_with_ranges)} COLUMNS mit Ranges/Listen gefunden")
        print("\n📊 Beispiele vor Expansion:")
        for idx, row in columns_with_ranges.head(3).iterrows():
            print(f"  • {row['Table'][:40]}... - Column: '{row['Column']}'")
        
        print("\n🔄 Expandiere Ranges/Listen für COLUMNS...")
        excerpts_df_columns = expand_column_ranges(excerpts_df_columns, column_name='Column')
        
        cols_after = len(excerpts_df_columns)
        print(f"✓ COLUMNS Expansion: {cols_before} → {cols_after} Zeilen (+{cols_after - cols_before})")
    else:
        print("✓ Keine Ranges/Listen in COLUMNS gefunden")
    
    # STEP 5: TABLE_SHORT SPALTE
    print("\n" + "="*60)
    print("STEP 5: Table_Short Spalte für Matching erstellen")
    print("="*60)
    
    excerpts_df_rows['Table_Short'] = excerpts_df_rows['Table'].apply(extract_table_code_prefix)
    excerpts_df_columns['Table_Short'] = excerpts_df_columns['Table'].apply(extract_table_code_prefix)
    
    print(f"✓ Table_Short Spalte hinzugefügt")
    
    # Preview
    print("\n📊 Preview Table_Short (erste 5):")
    unique_mappings = excerpts_df_rows[['Table', 'Table_Short']].drop_duplicates().head(5)
    for _, row in unique_mappings.iterrows():
        print(f"  • '{row['Table'][:40]}...' → '{row['Table_Short']}'")
    
    # STEP 6: BEREINIGUNG UND COMPONENT LABEL
    print("\n" + "="*60)
    print("STEP 6: Bereinigung und ComponentLabel hinzufügen")
    print("="*60)
    
    # 6a: Zeilen löschen wo "Text ohne Überschrift" leer ist ODER Row/Column leer ist
    print("🧹 Lösche Zeilen mit leerem 'Text ohne Überschrift' oder leerem Row/Column...")
    rows_before_clean = len(excerpts_df_rows)
    cols_before_clean = len(excerpts_df_columns)
    
    # Hilfsfunktion für robuste Validierung
    def is_valid_value(val):
        """Prüft ob ein Wert gültig ist (nicht leer, nicht NaN, nicht nur Whitespace)"""
        if pd.isna(val):
            return False
        val_str = str(val).strip()
        if val_str == '' or val_str.lower() == 'nan' or val_str.lower() == 'none':
            return False
        return True
    
    # ROWS bereinigen
    excerpts_df_rows = excerpts_df_rows[
        excerpts_df_rows['Text ohne Überschrift'].apply(is_valid_value) &
        excerpts_df_rows['Row'].apply(is_valid_value)
    ].copy()
    
    # COLUMNS bereinigen
    excerpts_df_columns = excerpts_df_columns[
        excerpts_df_columns['Text ohne Überschrift'].apply(is_valid_value) &
        excerpts_df_columns['Column'].apply(is_valid_value)
    ].copy()
    
    rows_after_clean = len(excerpts_df_rows)
    cols_after_clean = len(excerpts_df_columns)
    
    print(f"✓ ROWS: {rows_before_clean} → {rows_after_clean} (-{rows_before_clean - rows_after_clean})")
    print(f"✓ COLUMNS: {cols_before_clean} → {cols_after_clean} (-{cols_before_clean - cols_after_clean})")
    
    # 6b: Trees laden
    print("\n🌳 Lade Baumstrukturen...")
    
    with open(tree_path, 'rb') as f:
        all_trees = pickle.load(f)
    
    print(f"✓ {len(all_trees)} Baumstrukturen geladen")
    
    # Zeige verfügbare Trees
    print("\n📊 Verfügbare Trees (erste 5):")
    for i, (table_code, comp_type) in enumerate(list(all_trees.keys())[:5]):
        print(f"  {i+1}. {table_code} | {comp_type}")
    
    # 6c: ComponentLabel für ROWS hinzufügen
    print("\n🔧 Füge ComponentLabel für ROWS hinzu...")
    excerpts_df_rows['ComponentLabel'] = excerpts_df_rows.apply(
        lambda row: get_component_label(
            row['Table_Short'], 
            row['Row'], 
            all_trees, 
            'Table row'
        ), 
        axis=1
    )
    
    label_found_rows = (excerpts_df_rows['ComponentLabel'] != 'N/A').sum()
    print(f"✓ ComponentLabel hinzugefügt: {label_found_rows}/{len(excerpts_df_rows)} gefunden")
    
    # 6d: ComponentLabel für COLUMNS hinzufügen
    print("\n🔧 Füge ComponentLabel für COLUMNS hinzu...")
    excerpts_df_columns['ComponentLabel'] = excerpts_df_columns.apply(
        lambda row: get_component_label(
            row['Table_Short'], 
            row['Column'], 
            all_trees, 
            'Table column'
        ), 
        axis=1
    )
    
    label_found_cols = (excerpts_df_columns['ComponentLabel'] != 'N/A').sum()
    print(f"✓ ComponentLabel hinzugefügt: {label_found_cols}/{len(excerpts_df_columns)} gefunden")
    
    # 6e: Zeilen mit ComponentLabel = "N/A" löschen
    print("\n🧹 Lösche Zeilen mit ComponentLabel = 'N/A'...")
    rows_before_na = len(excerpts_df_rows)
    cols_before_na = len(excerpts_df_columns)
    
    excerpts_df_rows = excerpts_df_rows[excerpts_df_rows['ComponentLabel'] != 'N/A'].copy()
    excerpts_df_columns = excerpts_df_columns[excerpts_df_columns['ComponentLabel'] != 'N/A'].copy()
    
    rows_after_na = len(excerpts_df_rows)
    cols_after_na = len(excerpts_df_columns)
    
    print(f"✓ ROWS: {rows_before_na} → {rows_after_na} (-{rows_before_na - rows_after_na})")
    print(f"✓ COLUMNS: {cols_before_na} → {cols_after_na} (-{cols_before_na - cols_after_na})")
    
    # STEP 7: ALS CSV SPEICHERN
    print("\n" + "="*60)
    print("STEP 7: Ergebnisse speichern")
    print("="*60)
    
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    rows_filename = output_path / f"corep_annex_ROWS_{timestamp}.csv"
    columns_filename = output_path / f"corep_annex_COLUMNS_{timestamp}.csv"
    
    excerpts_df_rows.to_csv(rows_filename, index=False, encoding='utf-8-sig')
    print(f"✓ ROWS gespeichert: {rows_filename}")
    
    excerpts_df_columns.to_csv(columns_filename, index=False, encoding='utf-8-sig')
    print(f"✓ COLUMNS gespeichert: {columns_filename}")
    
    # ZUSAMMENFASSUNG
    print("\n" + "="*60)
    print("✅ COREP ANNEX EXTRACTION - ERFOLGREICH ABGESCHLOSSEN")
    print("="*60)
    print(f"\n📊 Zusammenfassung:")
    print(f"  • {len(content_dict)} Tabellen verarbeitet")
    print(f"  • {len(crr_refs)} CRR-Artikel")
    print(f"  • {len(excerpts_df_rows)} ROWS (bereinigt, mit ComponentLabel)")
    print(f"  • {len(excerpts_df_columns)} COLUMNS (bereinigt, mit ComponentLabel)")
    print("="*60)
    
    return excerpts_df_rows, excerpts_df_columns


if __name__ == "__main__":
    print("Starte COREP Annex Extraktion...")
    extract_corep_annex()