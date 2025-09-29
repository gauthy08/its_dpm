from sqlalchemy import create_engine, Column, Integer, String, Boolean, Date, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.orm import aliased
from sqlalchemy import text

import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from database.models import * 
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import os

# Lokale Module
from database.db_manager import SessionLocal, create_tables
from database.models import Template_Finrep
    
import pickle
import re


import pandas as pd
import pickle
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


from pathlib import Path
    

        
        
def find_correct_membername_for_reference():
    session = SessionLocal()
    try:
        # Abfrage: SELECT * FROM Konzept WHERE code = 'ISFIN0000001'
        entries = session.query(MergedData).all()
        data = []
        for entry in entries:
            row = entry.__dict__
            row.pop("_sa_instance_state", None)
            data.append(row)
        df_merge = pd.DataFrame(data)
        print(df_merge.head())
        print(df_merge.shape)
        #return df_merge
    finally:
        session.close()
    
    
    
    
    
    # 1. Fuzzy Similarity (Zeichen-basierter Vergleich)
    df_merge['fuzzy_similarity'] = df_merge.apply(
        lambda row: fuzz.ratio(row['member_name'], row['y_axis_name']), axis=1
    )

    # 2. Token-basierte Similarity (z. B. mit token_sort_ratio)
    df_merge['token_similarity'] = df_merge.apply(
        lambda row: fuzz.token_sort_ratio(row['member_name'], row['y_axis_name']), axis=1
    )

    # 3. TF-IDF-basierte Cosinus-Ähnlichkeit
    def tfidf_cosine_similarity(s1, s2):
        try:
            # Vektorisieren beider Strings
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([s1, s2])
            # Cosinus-Ähnlichkeit zwischen den beiden TF-IDF-Vektoren
            cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
            return cos_sim[0][0]*100
        except Exception as e:
            return np.nan

    #df_merge['tfidf_similarity'] = df_merge.apply(
     #   lambda row: tfidf_cosine_similarity(row['member_name'], row['y_axis_name']), axis=1
    #)

    # 4. Semantische Similarity mit Sentence Transformers
    # Lade ein vortrainiertes Modell (z.B. all-MiniLM-L6-v2)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def semantic_similarity(s1, s2):
        try:
            # Berechne die Embeddings der beiden Texte
            emb1 = model.encode(s1)
            emb2 = model.encode(s2)
            # Cosinus-Ähnlichkeit zwischen den Embeddings
            cos_sim = cosine_similarity([emb1], [emb2])
            return cos_sim[0][0]*100
        except Exception as e:
            return np.nan

    #df_merge['semantic_similarity'] = df_merge.apply(
      #  lambda row: semantic_similarity(row['member_name'], row['y_axis_name']), axis=1
    #)
    
    
    # 1. Durchschnitt der vier Ähnlichkeitswerte pro Zeile berechnen
    #similarity_cols = ["fuzzy_similarity", "token_similarity", "tfidf_similarity", "semantic_similarity"]
    similarity_cols = ["fuzzy_similarity", "token_similarity"]

    df_merge["avg_similarity"] = df_merge[similarity_cols].mean(axis=1)

    # 2. Neue Spalte Y_Winner initialisieren (zunächst mit NaN)
    df_merge["Y_Winner"] = np.nan

    # 3. Pro konzept_code die Zeile mit dem höchsten avg_similarity ermitteln und Y_Winner setzen
    for code, group in df_merge.groupby("konzept_code"):
        # Index der Zeile mit dem höchsten Durchschnittswert in dieser Gruppe
        max_idx = group["avg_similarity"].idxmax()
        # In dieser Zeile wird in Simil_Winner der Durchschnittswert geschrieben
        df_merge.loc[max_idx, "Simil_Winner"] = df_merge.loc[max_idx, "avg_similarity"]
    
    
    
    
    #######Plausi ausgaben#####
    
    # Anzahl der eindeutigen konzept_code
    unique_konzept_codes = df_merge['konzept_code'].nunique()

    # Anzahl der Zeilen, bei denen Y_Winner > 90 ist
    y_winner_over_90 = df_merge[df_merge['Simil_Winner'] > 60]

    print("Anzahl unique konzept_code:", unique_konzept_codes)
    print("Anzahl der Simil_Winner über 60:", y_winner_over_90.shape[0])

    y_winner_under_90 = df_merge[df_merge['Simil_Winner'] < 60]
    #pd.set_option('display.max_rows', 1)
    y_winner_under_90[['konzept_code', 'y_axis_rc_code', 'member_name', 'y_axis_name', 'Simil_Winner']].to_csv("output2.csv", index=False)
    
    
    #####Ideen: of which entfernen? 
    ##########x Achse hinzufügen
    ########abweichungen kontrollieren 
    ##########################
    
    
    df_merge.to_csv("output.csv", index=False)
    
    
    
    
    ############import reference 
    
    df_ref = match_merge_with_reference() 
    print(df_ref.head(1))





########################## C O R E P ########################

import pandas as pd
import pickle
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Diese Funktionen müssen Sie aus Ihrem bestehenden Code einbinden:
# from your_module import ITSBaseData_new, find_node_in_tree

def get_its(taxonomy_code: str):
    """Lädt ITS-Daten aus der Datenbank"""
    DATABASE_URL = "sqlite:///database.db"
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    
    session = Session()
    try:
        entries = (
            session
            .query(ITSBaseData_new)
            .filter(ITSBaseData_new.taxonomy_code == taxonomy_code)
            .all())
        
        data = [entry.__dict__ for entry in entries]
        df = pd.DataFrame(data).drop(columns=["_sa_instance_state"], errors='ignore')
        
        # Zuerst nach ko sortieren
        df_sorted = df.sort_values(by='ko', ascending=True)
        
        # Dann Duplikate in ko entfernen
        df_unique = df_sorted.drop_duplicates(subset='ko', keep='first')
        
        # Ausgabe der eindeutigen table_codes
        table_codes_sorted = sorted(df_unique['table_code'].dropna().unique())
        print("Eindeutige table_codes (aufsteigend sortiert):")
        print(table_codes_sorted)
        
        return df_unique[['ko', 'y_axis_rc_code', 'table_code', 'y_axis_name', 'x_axis_name', 'x_axis_rc_code']]
    finally:
        session.close()

def extract_table_code_prefix(full_table_name):
    """
    Extrahiert den Tabellen-Code vor dem ersten " -" aus einem vollständigen Tabellennamen
    z.B. "C 05.01 - TRANSITIONAL PROVISIONS (CA5.1)" -> "C 05.01"
    """
    if pd.isna(full_table_name) or str(full_table_name) == "nan":
        return ""
    
    full_name = str(full_table_name).strip()
    if " -" in full_name:
        return full_name.split(" -")[0].strip()
    else:
        return full_name

def find_node_by_componentcode(node, componentcode):
    """Findet einen Knoten anhand des componentcode"""
    if node.componentcode == componentcode:
        return node
    for child in node.children:
        result = find_node_by_componentcode(child, componentcode)
        if result:
            return result
    return None

def find_node_in_tree(roots, componentcode):
    """Findet einen Knoten in einem Baum anhand des componentcode"""
    if roots is None:
        return None
    for root in roots:
        result = find_node_by_componentcode(root, componentcode)
        if result:
            return result
    return None

def load_rag_texts(axis):
    """
    RAG-Texte für eine Achse laden
    """
    if axis == "x":
        filename = "Corep Rag/COREP_Chatbot_Columns.xlsx"
        code_column = "Column"
    else:  # axis == "y"
        filename = "Corep Rag/COREP_Chatbot_Rows.xlsx"
        code_column = "Row"
        
    coord_to_rag = {}
    
    try:
        if os.path.exists(filename):
            df_text = pd.read_excel(filename, sheet_name="Sheet1", dtype=str)
            
            print(f"RAG-Datei geladen: {filename} ({len(df_text)} Zeilen)")
            
            for _, row in df_text.iterrows():
                # Table Code extrahieren (kurze Version)
                full_table_name = str(row["Table"]) if "Table" in df_text.columns and pd.notnull(row["Table"]) else "UNKNOWN"
                table_code = extract_table_code_prefix(full_table_name)
                
                # Code normalisieren (mit führenden Nullen auf 4 Stellen)
                coord_code = str(row[code_column]).zfill(4) if code_column in df_text.columns and pd.notnull(row[code_column]) else "0000"
                
                # Schlüssel erstellen: (table_code, axis, coord_code)
                key = (table_code, axis, coord_code)
                
                # RAG-Text aus ChatBot-Spalte
                chatbot_text = str(row["ChatBot"]) if "ChatBot" in df_text.columns and pd.notnull(row["ChatBot"]) and str(row["ChatBot"]) != "nan" else ""
                
                # Referenzen aus Legal References
                legal_ref = str(row["Legal References"]) if "Legal References" in df_text.columns and pd.notnull(row["Legal References"]) and str(row["Legal References"]) != "nan" else ""
                
                # Nur gültige Einträge hinzufügen
                if chatbot_text:
                    entry = {
                        "text": chatbot_text,
                        "reference": legal_ref
                    }
                    
                    coord_to_rag.setdefault(key, []).append(entry)

            print(f"RAG-Texte für {axis.upper()}-Achse erfolgreich geladen ({len(coord_to_rag)} Einträge)")
                
        else:
            print(f"Warnung: RAG-Text-Datei {filename} nicht gefunden.")
    except Exception as e:
        print(f"Fehler beim Laden der {axis.upper()}-Achse RAG-Texte: {e}")
        import traceback
        traceback.print_exc()
        
    return coord_to_rag

def enrich_tree_with_rag_texts(trees, coord_to_rag_y, coord_to_rag_x, missing_rag_nodes=None):
    """
    Anreichern der Baumstruktur mit RAG-Texten (EINMALIG!)
    """
    enriched_trees = {}
    
    for (table_code, comp_type), roots in trees.items():
        enriched_roots = []
        axis = "y" if comp_type == "Table row" else "x"
        coord_to_rag = coord_to_rag_y if axis == "y" else coord_to_rag_x
        
        for root in roots:
            enriched_root = enrich_node_recursive(root, table_code, axis, coord_to_rag, missing_rag_nodes)
            enriched_roots.append(enriched_root)
            
        enriched_trees[(table_code, comp_type)] = enriched_roots
    
    return enriched_trees

def enrich_node_recursive(node, table_code, axis, coord_to_rag, missing_rag_nodes=None):
    """
    Rekursive Anreicherung eines Knotens mit RAG-Texten
    """
    # Node-Code aus componentcode holen
    node_code = str(node.componentcode).zfill(4)
    
    # Kurzer Table-Code für besseres Matching
    short_table_code = extract_table_code_prefix(table_code)
    
    # RAG-Text-Suche mit korrekter Schlüssel-Bildung
    key = (short_table_code, axis, node_code)
    rag_texts = coord_to_rag.get(key, [])
    
    # Falls kein direkter Match, versuche andere verfügbare Table-Codes mit gleichem Code
    if not rag_texts:
        for (existing_table, existing_axis, existing_code), rag_data in coord_to_rag.items():
            if existing_axis == axis and existing_code == node_code:
                # Versuche Partial-Match des Table-Codes
                if (short_table_code in existing_table or 
                    existing_table in short_table_code or
                    extract_table_code_prefix(existing_table) == short_table_code):
                    rag_texts = rag_data
                    break
    
    # RAG-Text formatieren
    if rag_texts:
        formatted_rag = ", ".join([
            f'{item["text"]}  \n-->Note: See {item["reference"]} for additional details and definitions.'
            if item["reference"] and item["reference"] != "nan" and item["reference"] != "" else item["text"]
            for item in rag_texts
        ])
    else:
        formatted_rag = " - "
        
        # Fehlenden RAG-Text sammeln (minimal invasiv hinzugefügt)
        if missing_rag_nodes is not None:
            missing_entry = {
                'Table': short_table_code,
                'Row_or_Column': node_code,
                'ComponentLabel': getattr(node, 'componentlabel', 'Unknown') if hasattr(node, 'componentlabel') else 'Unknown',
                'Axis': axis.upper(),
                'Level': getattr(node, 'level', 'Unknown') if hasattr(node, 'level') else 'Unknown',
                'LookupKey': f"({short_table_code}, {axis}, {node_code})"
            }
            missing_rag_nodes.append(missing_entry)
    
    # Knoten erweitern
    node.rag_text = formatted_rag
    
    # Kinder rekursiv bearbeiten
    if hasattr(node, 'children') and node.children:
        for child in node.children:
            enrich_node_recursive(child, table_code, axis, coord_to_rag, missing_rag_nodes)
    
    return node

def create_output_corep():
    print("hello Corep (optimiert)")

    # GEWÜNSCHTE TEMPLATE CODES - NUR DIESE WERDEN VERARBEITET
    allowed_template_codes = [
        "C 01.00", "C 02.00", "C 03.00", "C 04.00", 
        "C 05.01", "C 05.02", 
        "C 06.01", "C 06.02", 
        "C 07.00", 
        "C 08.01", "C 08.02", "C 08.03", "C 08.04", "C 08.05", "C 08.05.1", "C 08.06", "C 08.07", 
        "C 09.01", "C 09.02", "C 09.04", 
        "C 10.01", "C 10.02", "C 11.00", "C 13.01", "C 14.00", "C 14.01", 
        "C 34.01", "C 34.02", "C 34.03", "C 34.04", "C 34.05", "C 34.06", "C 34.07", "C 34.08", "C 34.09", "C 34.10", "C 34.11", 
        "C 16.00", 
        "C 17.01", "C 17.02"
    ]
    
    print(f"Verarbeitung beschränkt auf {len(allowed_template_codes)} Template Codes:")
    print(", ".join(allowed_template_codes))

    # Liste für fehlende RAG-Texte (minimal invasiv hinzugefügt)
    missing_rag_nodes = []

    # SCHRITT 1: Baumstruktur laden
    with open("baumstruktur_COREP_3_2.pkl", "rb") as f:
        trees = pickle.load(f)
    print("COREP-Baumstruktur geladen")

    # SCHRITT 2: RAG-Texte laden
    coord_to_rag_y = load_rag_texts("y")
    coord_to_rag_x = load_rag_texts("x")
    
    # SCHRITT 3: Baumstruktur mit RAG-Texten anreichern (EINMALIG!)
    enriched_trees = enrich_tree_with_rag_texts(trees, coord_to_rag_y, coord_to_rag_x, missing_rag_nodes)
    print("Baumstruktur mit RAG-Texten angereichert")

    # SCHRITT 4: ITS-Daten laden
    its_full = get_its("COREP_3.2")
    print("ITS-Daten geladen")
    
    # SCHRITT 4.1: ITS-Daten auf gewünschte Template Codes filtern
    its = its_full[its_full['table_code'].isin(allowed_template_codes)].copy()
    print(f"ITS-Daten gefiltert: {len(its_full)} -> {len(its)} Zeilen (nur gewünschte Template Codes)")
    
    # Überprüfe welche Template Codes tatsächlich vorhanden sind
    found_codes = sorted(its['table_code'].unique())
    missing_codes = set(allowed_template_codes) - set(found_codes)
    print(f"Gefundene Template Codes ({len(found_codes)}): {', '.join(found_codes)}")
    if missing_codes:
        print(f"Fehlende Template Codes ({len(missing_codes)}): {', '.join(sorted(missing_codes))}")

    # SCHRITT 5: ITS-Zeilen mit angereicherten Bäumen verarbeiten
    matches_found = {"y": 0, "x": 0}
    rag_matches_found = {"y": 0, "x": 0}
    
    for idx, row in its.iterrows():
        table_code = row['table_code']
        
        # Angereicherte Bäume finden
        tree_y = None
        tree_x = None
        
        its_short = extract_table_code_prefix(table_code)
        
        for (tree_table_code, comp_type), roots in enriched_trees.items():
            tree_short = extract_table_code_prefix(tree_table_code)
            
            # Mehrere Matching-Strategien versuchen
            match_found = False
            
            # Strategie 1: Exakte Übereinstimmung der kurzen Codes
            if tree_short == its_short:
                match_found = True
            # Strategie 2: ITS Code ist Teilstring des Baum-Codes
            elif its_short in tree_short:
                match_found = True
            # Strategie 3: Baum Code ist Teilstring des ITS-Codes  
            elif tree_short in its_short:
                match_found = True
            
            if match_found:
                if comp_type == "Table row":
                    tree_y = roots
                    matches_found["y"] += 1
                elif comp_type == "Table column":
                    tree_x = roots
                    matches_found["x"] += 1

        # Y-Achse verarbeiten
        rag_found_y = process_axis(its, idx, row, tree_y, 'y', 'y_axis_rc_code')
        if rag_found_y:
            rag_matches_found["y"] += 1
            
        # X-Achse verarbeiten  
        rag_found_x = process_axis(its, idx, row, tree_x, 'x', 'x_axis_rc_code')
        if rag_found_x:
            rag_matches_found["x"] += 1

        # Debug-Ausgabe für erste paar Zeilen
        if idx < 3:
            print(f"Debug Row {idx}: ITS='{its_short}' -> Y:{'✓' if tree_y else '✗'}(RAG:{'✓' if rag_found_y else '✗'}), X:{'✓' if tree_x else '✗'}(RAG:{'✓' if rag_found_x else '✗'})")

    print(f"\n=== MATCHING STATISTIK ===")
    print(f"Y-Achse Tree Matches: {matches_found['y']}, RAG Matches: {rag_matches_found['y']}")
    print(f"X-Achse Tree Matches: {matches_found['x']}, RAG Matches: {rag_matches_found['x']}")
    print(f"Gesamt ITS-Zeilen (gefiltert): {len(its)}")

    # Export der fehlenden RAG-Texte (minimal invasiv hinzugefügt)
    if missing_rag_nodes:
        df_missing = pd.DataFrame(missing_rag_nodes)
        # Duplikate entfernen (falls Knoten mehrfach verarbeitet wurden)
        df_missing = df_missing.drop_duplicates()
        df_missing = df_missing.sort_values(['Table', 'Axis', 'Row_or_Column'])
        
        # Nach Rows und Columns trennen
        df_rows = df_missing[df_missing['Axis'] == 'Y'][['Table', 'Row_or_Column', 'ComponentLabel']].copy()
        df_cols = df_missing[df_missing['Axis'] == 'X'][['Table', 'Row_or_Column', 'ComponentLabel']].copy()
        
        # Excel exportieren
        with pd.ExcelWriter("Missing_RAG_Texts.xlsx", engine="openpyxl") as writer:
            if not df_rows.empty:
                df_rows.to_excel(writer, sheet_name="Missing_Rows", index=False)
            if not df_cols.empty:
                df_cols.to_excel(writer, sheet_name="Missing_Columns", index=False)
        
        print(f"✓ Fehlende RAG-Texte exportiert: Missing_RAG_Texts.xlsx ({len(df_missing)} Einträge)")
        print(f"  - Fehlende Rows: {len(df_rows)}, Fehlende Columns: {len(df_cols)}")
    else:
        print("🎉 Alle Knoten haben RAG-Texte!")

    # SCHRITT 6: Finale Ausgabe
    finalize_output(its)
    return its

def process_axis(its, idx, row, tree, axis_name, rc_code_column):
    """
    Verarbeitung einer Achse mit angereicherten Bäumen
    """
    componentcode = row[rc_code_column]
    node = find_node_in_tree(tree, componentcode) if tree is not None else None

    # Spalten initialisieren falls nicht vorhanden
    if f'{axis_name}-found_in_tree' not in its.columns:
        its[f'{axis_name}-found_in_tree'] = None
    if f'{axis_name}-parent_path' not in its.columns:
        its[f'{axis_name}-parent_path'] = None

    rag_found = False

    if node:
        try:
            # Pfad extrahieren - verwenden die bestehenden bewährten Methoden
            labels = node.get_path_codes()
            levels = node.get_level_codes()  
            componentlabel = node.get_component_label()
            pairs = sorted(zip(levels, labels, componentlabel), key=lambda x: int(x[0]))

            # Pfad-String zusammenbauen MIT RAG-TEXTEN
            path_string_parts = []
            
            # Für jeden Level im Pfad den entsprechenden Knoten finden und RAG-Text verwenden
            for (lvl, code, comp) in pairs:
                # Finde den Knoten mit diesem Code im Baum
                code_node = find_node_in_tree(tree, code)
                
                if code_node and hasattr(code_node, 'rag_text'):
                    # RAG-Text direkt aus dem angereicherten Knoten holen
                    rag_text = code_node.rag_text
                    if rag_text != " - ":
                        rag_found = True
                else:
                    # Fallback: " - " wenn Knoten nicht gefunden oder nicht angereichert
                    rag_text = " - "
                
                # KORRIGIERTES FORMAT: Level, Component Label und RAG-Text
                part_str = f'***LEVEL {lvl}*** "{comp}": {rag_text}'
                path_string_parts.append(part_str)

            # Prefix und finale Pfad-Zusammenstellung
            if axis_name == 'y':
                prefix_str = (
                    f'***Y-AXIS NAME***: {row["y_axis_name"]}  \n'
                    f'***X-AXIS NAME***: {row["x_axis_name"]}\n\n'
                    '***Y-AXIS***:  \n'
                )
            else:
                prefix_str = '\n***X-AXIS***:  \n'
                
            parent_path_str = prefix_str + "  \n".join(path_string_parts)
            its.at[idx, f'{axis_name}-found_in_tree'] = True
            its.at[idx, f'{axis_name}-parent_path'] = parent_path_str
            
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von Knoten {componentcode}: {e}")
            its.at[idx, f'{axis_name}-found_in_tree'] = False
            its.at[idx, f'{axis_name}-parent_path'] = None
    else:
        its.at[idx, f'{axis_name}-found_in_tree'] = False
        its.at[idx, f'{axis_name}-parent_path'] = None
    
    return rag_found

def finalize_output(its):
    """
    Finale Ausgabe-Verarbeitung
    """
    disclaimer_text = "\n\n*Disclaimer: This AI-generated output is for guidance only and may require further review. Please consult the referenced materials and perform your own research to ensure accuracy.*"
    
    def combine_paths(row):
        y_path = row['y-parent_path'] if pd.notnull(row['y-parent_path']) else ""
        x_path = row['x-parent_path'] if pd.notnull(row['x-parent_path']) else ""
        combined = y_path + "\n" + x_path if y_path or x_path else ""
        return combined + disclaimer_text if combined else disclaimer_text

    its['combined_path'] = its.apply(combine_paths, axis=1)

    # Debug-Ausgabe
    pd.set_option('display.max_colwidth', None)
    print("\n--- Beispiel output ---")
    if len(its) > 0:
        print(f"KO: {its.iloc[0]['ko']}")
        print("Combined Path:")
        sample_text = its.iloc[0]['combined_path']
        print(sample_text[:1000] + "..." if len(sample_text) > 1000 else sample_text)

    # Ausgabe
    its.to_excel("output.xlsx", index=False, engine="openpyxl")
    
    # Excel-Datei erstellen
    try:
        if os.path.exists("Files/GemKonz_Vorlage.xlsx"):
            df_excel = pd.read_excel("Files/GemKonz_Vorlage.xlsx", engine="openpyxl")
        else:
            print("Warnung: Datei 'Files/GemKonz_Vorlage.xlsx' nicht gefunden. Erstelle neue DataFrame.")
            df_excel = pd.DataFrame()
            
        df_excel["Code"] = its["ko"]
        df_excel["Beschreibung"] = its["combined_path"]
        
        with pd.ExcelWriter("Corep_output.xlsx", engine="openpyxl", mode="w") as writer:
            df_excel.to_excel(writer, index=False, sheet_name="Sheet1")

        print("Excel-Datei 'Corep_output.xlsx' erfolgreich erstellt.")
        
    except Exception as e:
        print(f"Fehler beim Erstellen der Excel-Datei: {e}")

# WICHTIGER HINWEIS: 
# Diese Funktion muss mit Ihrem bestehenden Code integriert werden:
# from your_database_models import ITSBaseData_new
# Ersetzen Sie "your_database_models" durch den tatsächlichen Modulnamen

# WICHTIGER HINWEIS: 
# Diese Funktion muss mit Ihrem bestehenden Code integriert werden:
# from your_database_models import ITSBaseData_new
# Ersetzen Sie "your_database_models" durch den tatsächlichen Modulnamen


########################## F I N R E P ########################
def create_output():
    print("hello world11")

    # Load ITS data
    its = get_its()
    #print("ITS HEAD: ", its.head())
    
    sorted_table_ids = sorted(its["table_id"].dropna().unique())

    # --- y-Achse (RAG-Texte) ---
    df_y_text = pd.read_excel("finrep_references/df_enriched_y_axis13_5-2025.xlsx", sheet_name="Clean", dtype=str)
    df_y_text["Coord"] = df_y_text["Coord"].str.zfill(4)  # <--- FIX HIER


    coord_to_rag_y = {}
    for _, row in df_y_text.iterrows():
        key = (row["Worksheet_new"], row["Axis"], row["Coord"])
        entry = {
            "text": str(row["RAG_Text"]),
            "reference": str(row["Reference"])
        }
        coord_to_rag_y.setdefault(key, []).append(entry)



    y_table = df_y_text["Worksheet_new"].iloc[0]

    # --- x-Achse (RAG-Texte) ---
    df_text = pd.read_excel("finrep_references/df_enriched_x_axis13_5_2025.xlsx", sheet_name="Clean", dtype=str)
    df_text["Coord"] = df_text["Coord"].str.zfill(4)  # <--- FIX HIER


    coord_to_rag_x = {}
    for _, row in df_text.iterrows():
        key = (row["Worksheet_new"], row["Axis"], row["Coord"])
        entry = {
            "text": str(row["RAG_Text"]),
            "reference": str(row["Reference"])
        }
        coord_to_rag_x.setdefault(key, []).append(entry)


    x_table = df_text["Worksheet_new"].iloc[0]

    # Load tree structure
    with open("baumstruktur.pkl", "rb") as f:
        trees = pickle.load(f)

    for idx, row in its.iterrows():
        table_id = row['table_id']
        tree_y = None
        tree_x = None

        for (table_code, comp_type), roots in trees.items():
            if table_code.startswith(table_id):
                if comp_type == "Table row":
                    tree_y = roots
                elif comp_type == "Table column":
                    tree_x = roots

        # --- y-Achse ---
        componentcode_y = row['y_axis_rc_code']
        node_y = find_node_in_tree(tree_y, componentcode_y) if tree_y is not None else None

        if node_y:
            labels_y = node_y.get_path_codes()
            levels_y = node_y.get_level_codes()
            componentlabel_y = node_y.get_component_label()
            pairs_y = sorted(zip(levels_y, labels_y, componentlabel_y), key=lambda x: int(x[0]))

            path_string_parts_y = []
            for (lvl, code, comp) in pairs_y:
                rag_texts = coord_to_rag_y.get((row["table_id"], "y", code), None)
                #rag_text = ", ".join([f'{item["text"]} [{item["reference"]}]' for item in rag_texts]) if rag_texts else " - "
                rag_text = ", ".join([
                    #f'{item["text"]} [relates to {item["reference"]}]'
                    f'{item["text"]}  \n-->Note: See {item["reference"]} for additional details and definitions.'
                    if item["reference"] else item["text"]
                    for item in rag_texts
                ]) if rag_texts else " - "
                part_str = f'***LEVEL {lvl}*** "{comp}": {rag_text}'
                path_string_parts_y.append(part_str)

            prefix_str = (
                f'***Y-AXIS NAME***: {row["y_axis_name"]}  \n'
                f'***X-AXIS NAME***: {row["x_axis_name"]}\n\n'
                '***Y-AXIS***:  \n'
            )
            parent_path_str_y = prefix_str + "  \n".join(path_string_parts_y)
            its.at[idx, 'y-found_in_tree'] = True
            its.at[idx, 'y-parent_path'] = parent_path_str_y
        else:
            its.at[idx, 'y-found_in_tree'] = False
            its.at[idx, 'y-parent_path'] = None

        # --- x-Achse ---
        componentcode_x = row['x_axis_rc_code']
        node_x = find_node_in_tree(tree_x, componentcode_x) if tree_x is not None else None

        if node_x:
            labels_x = node_x.get_path_codes()
            levels_x = node_x.get_level_codes()
            componentlabel_x = node_x.get_component_label()
            pairs_x = sorted(zip(levels_x, labels_x, componentlabel_x), key=lambda x: int(x[0]))

            path_string_parts_x = []
            for (lvl, code, comp) in pairs_x:
                rag_texts = coord_to_rag_x.get((row["table_id"], "x", code), None)
                #rag_text = ", ".join([f'{item["text"]} [{item["reference"]}]' for item in rag_texts]) if rag_texts else " - "
                rag_text = ", ".join([
                    #f'{item["text"]} [relates to {item["reference"]}]'
                    f'{item["text"]}  \n-->Note: See {item["reference"]} for additional details and definitions.'
                    if item["reference"] else item["text"]
                    for item in rag_texts
                ]) if rag_texts else " - "
                part_str = f'***LEVEL {lvl}*** "{comp}": {rag_text}'
                path_string_parts_x.append(part_str)

            prefix_str_x = '\n***X-AXIS***:  \n'
            parent_path_str_x = prefix_str_x + "  \n".join(path_string_parts_x)
            its.at[idx, 'x-found_in_tree'] = True
            its.at[idx, 'x-parent_path'] = parent_path_str_x
        else:
            its.at[idx, 'x-found_in_tree'] = False
            its.at[idx, 'x-parent_path'] = None

    # --- Kombiniere y- und x-Pfade ---
    disclaimer_text = "\n\n*Disclaimer: This AI-generated output is for guidance only and may require further review. Please consult the referenced materials and perform your own research to ensure accuracy.*"
    
    def combine_paths(row):
        y_path = row['y-parent_path'] if pd.notnull(row['y-parent_path']) else ""
        x_path = row['x-parent_path'] if pd.notnull(row['x-parent_path']) else ""
        combined = y_path + "\n" + x_path if y_path or x_path else ""
        return combined + disclaimer_text if combined else disclaimer_text

    its['combined_path'] = its.apply(combine_paths, axis=1)

    pd.set_option('display.max_colwidth', None)

    print("\n--- Beispiel output ---")
    print(its[['konzept_code', 'combined_path']].head(1))

    # Ausgabe
    its.to_excel("output.xlsx", index=False, engine="openpyxl")

    df_excel = pd.read_excel("Files/GemKonz_mas_leer.xlsx", engine="openpyxl")
    df_excel["Code"] = its["konzept_code"]
    df_excel["Beschreibung"] = its["combined_path"]
    #df_excel.to_excel("GemKonz_output.xlsx", index=False, engine="openpyxl")
    
    
    # Mit expliziter Kodierungsbehandlung speichern
    with pd.ExcelWriter("GemKonz_output.xlsx", engine="openpyxl", mode="w") as writer:
        df_excel.to_excel(writer, index=False, sheet_name="Sheet1")
        # Sicherstellen, dass die Arbeitsmappe mit der richtigen Kodierung gespeichert wird
        writer.book.encoding = "utf-8"

        # Optional: Spaltenbreite anpassen, damit alle Zeichen sichtbar sind
        #for column in df_excel:
        #    column_width = max(df_excel[column].astype(str).map(len).max(), len(column)) + 2
        #    col_idx = df_excel.columns.get_loc(column)
        #    writer.sheets["Sheet1"].column_dimensions[chr(65 + col_idx)].width = column_width

    return its

def createUpload():
    """
    Neue Funktion die ähnlich wie create_output_corep() funktioniert,
    aber RAG-Texte aus production run files bezieht.
    """
    print("=== CREATE UPLOAD - Production Run zu GemKonz ===")
    
    # SCHRITT 1: .pkl File aus tree_structures auswählen
    tree_dir = Path("tree_structures")
    if not tree_dir.exists():
        print("❌ Ordner 'tree_structures' nicht gefunden!")
        return
    
    pickle_files = list(tree_dir.glob("*.pkl"))
    if not pickle_files:
        print("❌ Keine .pkl Dateien in 'tree_structures' gefunden!")
        return
    
    print(f"\n📁 Verfügbare Pickle Files:")
    for i, file in enumerate(pickle_files, 1):
        print(f"   {i}. {file.name}")
    
    try:
        choice = int(input(f"\nWähle Pickle File (1-{len(pickle_files)}): ")) - 1
        if not (0 <= choice < len(pickle_files)):
            print("❌ Ungültige Auswahl")
            return
        selected_pickle = pickle_files[choice]
    except ValueError:
        print("❌ Bitte eine gültige Zahl eingeben")
        return
    
    # SCHRITT 2: Taxonomy Code auswählen
    available_taxonomies = ["FINREP_3.2.1", "COREP_3.2"]
    print(f"\n🏷️ Verfügbare Taxonomy Codes:")
    for i, tax in enumerate(available_taxonomies, 1):
        print(f"   {i}. {tax}")
    
    try:
        tax_choice = int(input(f"\nWähle Taxonomy Code (1-{len(available_taxonomies)}): ")) - 1
        if not (0 <= tax_choice < len(available_taxonomies)):
            print("❌ Ungültige Auswahl")
            return
        selected_taxonomy = available_taxonomies[tax_choice]
    except ValueError:
        print("❌ Bitte eine gültige Zahl eingeben")
        return
    
    # SCHRITT 3: Production Run File auswählen
    prod_dir = Path("evaluation_experiments/production_runs")
    if not prod_dir.exists():
        print("❌ Ordner 'evaluation_experiments/production_runs' nicht gefunden!")
        return
    
    production_files = list(prod_dir.glob("*.xlsx"))
    if not production_files:
        print("❌ Keine .xlsx Dateien in 'evaluation_experiments/production_runs' gefunden!")
        return
    
    print(f"\n📊 Verfügbare Production Run Files:")
    for i, file in enumerate(production_files, 1):
        print(f"   {i}. {file.name}")
    
    try:
        prod_choice = int(input(f"\nWähle Production Run File (1-{len(production_files)}): ")) - 1
        if not (0 <= prod_choice < len(production_files)):
            print("❌ Ungültige Auswahl")
            return
        selected_production_file = production_files[prod_choice]
    except ValueError:
        print("❌ Bitte eine gültige Zahl eingeben")
        return
    
    print(f"\n✅ Ausgewählt:")
    print(f"   📁 Pickle File: {selected_pickle.name}")
    print(f"   🏷️ Taxonomy: {selected_taxonomy}")
    print(f"   📊 Production File: {selected_production_file.name}")
    
    # SCHRITT 4: Daten laden
    print(f"\n🔄 Lade Daten...")
    
    # Baumstruktur laden
    try:
        with open(selected_pickle, "rb") as f:
            trees = pickle.load(f)
        print(f"✅ Baumstruktur geladen: {len(trees)} tree combinations")
    except Exception as e:
        print(f"❌ Fehler beim Laden der Baumstruktur: {e}")
        return
    
    # Production Run Daten laden
    try:
        production_df = pd.read_excel(selected_production_file, sheet_name='Results')
        print(f"✅ Production Run Daten geladen: {len(production_df)} responses")
    except Exception as e:
        print(f"❌ Fehler beim Laden der Production Run Daten: {e}")
        return
    
    # ITS Daten laden
    try:
        its_full = get_its(selected_taxonomy)
        print(f"✅ ITS-Daten geladen: {len(its_full)} Zeilen")
    except Exception as e:
        print(f"❌ Fehler beim Laden der ITS-Daten: {e}")
        return
    
    # SCHRITT 5: RAG-Texte aus Production Run extrahieren
    rag_lookup = extract_rag_from_production_run(production_df)
    print(f"✅ RAG-Lookup erstellt: {len(rag_lookup)} Einträge")
    
    # SCHRITT 6: ITS-Daten mit RAG-Texten anreichern
    enriched_its = enrich_its_with_production_rag(its_full, trees, rag_lookup)
    
    # SCHRITT 7: Finale Ausgabe erstellen
    create_gemkonz_output(enriched_its, "Upload_GemKonz_output.xlsx")
    
    print(f"\n🎉 Upload-Pipeline abgeschlossen!")
    print(f"📁 Output: Upload_GemKonz_output.xlsx")
    
    return enriched_its


def extract_rag_from_production_run(production_df):
    """
    Extrahiert RAG-Texte aus Production Run Daten.
    Erwartet node_id Format: "table_code|component_type|node_code"
    """
    rag_lookup = {}
    
    for _, row in production_df.iterrows():
        try:
            # node_id parsen: "table_code|component_type|node_code"
            node_id = str(row['node_id'])
            parts = node_id.split('|')
            
            if len(parts) != 3:
                continue
                
            table_code, component_type, node_code = parts
            
            # Axis bestimmen
            axis = "y" if "row" in component_type.lower() else "x"
            
            # Kurzen Table Code extrahieren
            short_table_code = extract_table_code_prefix(table_code)
            
            # Key erstellen
            key = (short_table_code, axis, node_code.zfill(4))
            
            # RAG-Text aus chatbot_response
            rag_text = str(row.get('chatbot_response', ''))
            
            # Nur valide Antworten aufnehmen
            if rag_text and not rag_text.startswith(('API Error', 'Error')):
                rag_lookup[key] = rag_text
                
        except Exception as e:
            # Fehlerhafte Zeilen überspringen
            continue
    
    return rag_lookup


def enrich_its_with_production_rag(its_df, trees, rag_lookup):
    """
    Anreicherung der ITS-Daten mit Production Run RAG-Texten
    """
    its = its_df.copy()
    
    # Statistiken
    matches_found = {"y": 0, "x": 0}
    rag_matches_found = {"y": 0, "x": 0}
    
    for idx, row in its.iterrows():
        table_code = row['table_code']
        
        # Bäume finden (gleiche Logik wie in create_output_corep)
        tree_y = None
        tree_x = None
        
        its_short = extract_table_code_prefix(table_code)
        
        for (tree_table_code, comp_type), roots in trees.items():
            tree_short = extract_table_code_prefix(tree_table_code)
            
            # Matching-Strategien
            match_found = False
            
            if tree_short == its_short:
                match_found = True
            elif its_short in tree_short:
                match_found = True
            elif tree_short in its_short:
                match_found = True
            
            if match_found:
                if comp_type == "Table row":
                    tree_y = roots
                    matches_found["y"] += 1
                elif comp_type == "Table column":
                    tree_x = roots
                    matches_found["x"] += 1
        
        # Y-Achse verarbeiten
        rag_found_y = process_axis_with_production_rag(
            its, idx, row, tree_y, 'y', 'y_axis_rc_code', rag_lookup
        )
        if rag_found_y:
            rag_matches_found["y"] += 1
        
        # X-Achse verarbeiten
        rag_found_x = process_axis_with_production_rag(
            its, idx, row, tree_x, 'x', 'x_axis_rc_code', rag_lookup
        )
        if rag_found_x:
            rag_matches_found["x"] += 1
    
    print(f"\n=== MATCHING STATISTIK ===")
    print(f"Y-Achse Tree Matches: {matches_found['y']}, RAG Matches: {rag_matches_found['y']}")
    print(f"X-Achse Tree Matches: {matches_found['x']}, RAG Matches: {rag_matches_found['x']}")
    print(f"Gesamt ITS-Zeilen: {len(its)}")
    
    return its


def process_axis_with_production_rag(its, idx, row, tree, axis_name, rc_code_column, rag_lookup):
    """
    Verarbeitung einer Achse mit Production Run RAG-Texten
    """
    componentcode = row[rc_code_column]
    node = find_node_in_tree(tree, componentcode) if tree is not None else None
    
    # Spalten initialisieren
    if f'{axis_name}-found_in_tree' not in its.columns:
        its[f'{axis_name}-found_in_tree'] = None
    if f'{axis_name}-parent_path' not in its.columns:
        its[f'{axis_name}-parent_path'] = None
    
    rag_found = False
    
    if node:
        try:
            # Pfad extrahieren
            labels = node.get_path_codes()
            levels = node.get_level_codes()
            componentlabel = node.get_component_label()
            pairs = sorted(zip(levels, labels, componentlabel), key=lambda x: int(x[0]))
            
            # Pfad-String mit Production RAG-Texten
            path_string_parts = []
            table_short = extract_table_code_prefix(row['table_code'])
            
            for (lvl, code, comp) in pairs:
                # RAG-Text aus Production Run Lookup
                key = (table_short, axis_name, code.zfill(4))
                rag_text = rag_lookup.get(key, " - ")
                
                if rag_text != " - ":
                    rag_found = True
                
                # Format: Level, Component Label und RAG-Text
                part_str = f'***LEVEL {lvl}*** "{comp}": {rag_text}'
                path_string_parts.append(part_str)
            
            # Prefix erstellen
            if axis_name == 'y':
                prefix_str = (
                    f'***Y-AXIS NAME***: {row["y_axis_name"]}  \n'
                    f'***X-AXIS NAME***: {row["x_axis_name"]}\n\n'
                    '***Y-AXIS***:  \n'
                )
            else:
                prefix_str = '\n***X-AXIS***:  \n'
            
            parent_path_str = prefix_str + "  \n".join(path_string_parts)
            its.at[idx, f'{axis_name}-found_in_tree'] = True
            its.at[idx, f'{axis_name}-parent_path'] = parent_path_str
            
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von Knoten {componentcode}: {e}")
            its.at[idx, f'{axis_name}-found_in_tree'] = False
            its.at[idx, f'{axis_name}-parent_path'] = None
    else:
        its.at[idx, f'{axis_name}-found_in_tree'] = False
        its.at[idx, f'{axis_name}-parent_path'] = None
    
    return rag_found


def create_gemkonz_output(its_df, output_filename):
    """
    Erstellt die finale GemKonz-Output Datei
    """
    # Disclaimer hinzufügen
    disclaimer_text = "\n\n*Disclaimer: This AI-generated output is for guidance only and may require further review. Please consult the referenced materials and perform your own research to ensure accuracy.*"
    
    def combine_paths(row):
        y_path = row['y-parent_path'] if pd.notnull(row['y-parent_path']) else ""
        x_path = row['x-parent_path'] if pd.notnull(row['x-parent_path']) else ""
        combined = y_path + "\n" + x_path if y_path or x_path else ""
        return combined + disclaimer_text if combined else disclaimer_text
    
    its_df['combined_path'] = its_df.apply(combine_paths, axis=1)
    
    # Debug-Ausgabe
    print("\n--- Beispiel Output ---")
    if len(its_df) > 0:
        print(f"KO: {its_df.iloc[0]['ko']}")
        print("Combined Path:")
        sample_text = its_df.iloc[0]['combined_path']
        print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
    
    # Excel-Datei erstellen
    try:
        if os.path.exists("Files/GemKonz_Vorlage.xlsx"):
            df_excel = pd.read_excel("Files/GemKonz_Vorlage.xlsx", engine="openpyxl")
        else:
            print("Warnung: Datei 'Files/GemKonz_Vorlage.xlsx' nicht gefunden. Erstelle neue DataFrame.")
            df_excel = pd.DataFrame()
        
        df_excel["Code"] = its_df["ko"]
        df_excel["Beschreibung"] = its_df["combined_path"]
        
        with pd.ExcelWriter(output_filename, engine="openpyxl", mode="w") as writer:
            df_excel.to_excel(writer, index=False, sheet_name="Sheet1")
        
        print(f"✅ Excel-Datei '{output_filename}' erfolgreich erstellt.")
        
    except Exception as e:
        print(f"❌ Fehler beim Erstellen der Excel-Datei: {e}")
    
    return df_excel