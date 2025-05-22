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


# Lokale Module
from database.db_manager import SessionLocal, create_tables
from database.models import Konzept, ITSBaseData, DPM_datapoint, Template_Finrep, MergedData
    
import pickle
import re



def merge_data():
    session = SessionLocal()
    
    konzept_alias = aliased(Konzept)
    its_data_alias = aliased(ITSBaseData)

    query = session.query(
        konzept_alias.id.label('konzept_id'),
        its_data_alias.id.label('its_base_data_id'),
        its_data_alias.konzept_code,
        its_data_alias.datapoint,
        its_data_alias.taxonomy_code,
        its_data_alias.template_id,
        its_data_alias.template_label,
        its_data_alias.module_id,
        its_data_alias.module_gueltig_von,
        its_data_alias.module_gueltig_bis,
        its_data_alias.table_id,
        its_data_alias.table_name,
        its_data_alias.criteria,
        its_data_alias.x_axis_rc_code,
        its_data_alias.x_axis_name,
        its_data_alias.y_axis_rc_code,
        its_data_alias.y_axis_name,
        its_data_alias.z_axis_rc_code,
        its_data_alias.z_axis_name
    ).join(
        its_data_alias, konzept_alias.code == its_data_alias.konzept_code, isouter=True
    ).filter(
        its_data_alias.datapoint.isnot(None)  # Filtert Zeilen, in denen 'datapoint' NULL ist
    )

    # Ergebnisse in die neue Tabelle einfügen
    for row in query:
        merged_entry = MergedData(
            konzept_id=row.konzept_id,
            datapoint = row.datapoint,
            its_base_data_id=row.its_base_data_id,
            konzept_code=row.konzept_code,
            taxonomy_code=row.taxonomy_code,
            template_id=row.template_id,
            template_label=row.template_label,
            module_id=row.module_id,
            module_gueltig_von=row.module_gueltig_von,
            module_gueltig_bis=row.module_gueltig_bis,
            table_id=row.table_id,
            table_name=row.table_name,
            criteria=row.criteria,
            x_axis_rc_code=row.x_axis_rc_code,
            x_axis_name=row.x_axis_name,
            y_axis_rc_code=row.y_axis_rc_code,
            y_axis_name=row.y_axis_name,
            z_axis_rc_code=row.z_axis_rc_code,
            z_axis_name=row.z_axis_name
        )
        session.add(merged_entry)

    # Änderungen speichern
    session.commit()

    print("Left Merge erfolgreich durchgeführt und Daten gespeichert.")
    
    
def update_merged_data_with_dpm():
    print("Start creating expanded MergedData rows based on DPM_datapoint")
    session = SessionLocal()

    # Aliase für die Tabellen erstellen
    merged_alias = aliased(MergedData)
    dpm_alias = aliased(DPM_datapoint)

    # Left Join-Abfrage erstellen
    query = session.query(
        merged_alias,
        dpm_alias
    ).outerjoin(
        dpm_alias,
        merged_alias.datapoint == dpm_alias.datapoint_vid
    )

    # Für jede Zeile im Join Ergebnis legen wir einen neuen Eintrag in MergedData an
    for merged_row, dpm_row in query:
        # Falls es in DPM_datapoint keinen passenden Eintrag gibt, ist dpm_row=None
        new_entry = MergedData(
            # -- Felder aus dem "ursprünglichen" MergedData-Datensatz --
            konzept_id=merged_row.konzept_id,
            its_base_data_id=merged_row.its_base_data_id,
            datapoint=merged_row.datapoint,
            konzept_code=merged_row.konzept_code,
            taxonomy_code=merged_row.taxonomy_code,
            template_id=merged_row.template_id,
            template_label=merged_row.template_label,
            module_id=merged_row.module_id,
            module_gueltig_von=merged_row.module_gueltig_von,
            module_gueltig_bis=merged_row.module_gueltig_bis,
            table_id=merged_row.table_id,
            table_name=merged_row.table_name,
            criteria=merged_row.criteria,
            x_axis_rc_code=merged_row.x_axis_rc_code,
            x_axis_name=merged_row.x_axis_name,
            y_axis_rc_code=merged_row.y_axis_rc_code,
            y_axis_name=merged_row.y_axis_name,
            z_axis_rc_code=merged_row.z_axis_rc_code,
            z_axis_name=merged_row.z_axis_name,
            
            # -- Felder aus DPM_datapoint (oder None), wenn kein Match --
            datapoint_vid=dpm_row.datapoint_vid if dpm_row else None,
            dimension_label=dpm_row.dimension_label if dpm_row else None,
            member_name=dpm_row.member_name if dpm_row else None
        )
        session.add(new_entry)
    
    # Zunächst alle neuen Zeilen speichern
    session.commit()
    print("New rows inserted into MergedData.")

    # Anschließend alle Zeilen löschen, die den String 'None' in member_name haben
    delete_statement = text("DELETE FROM MergedData WHERE member_name IS NULL")
    session.execute(delete_statement)
    session.commit()
    
    print("Deleted all rows from MergedData where member_name IS NULL.")
    print("Finished. Neue Zeilen wurden in MergedData erzeugt.")

    
def match_merge_with_reference():
    session = SessionLocal()
    try:
        # Abfrage: SELECT * FROM MergedData ORDER BY id
        entries = session.query(Template_Finrep).all()
        data = []
        for entry in entries:
            row = entry.__dict__
            row.pop("_sa_instance_state", None)
            data.append(row)
        df = pd.DataFrame(data)
        print(df.head())
        print(df.shape)
        return df
    finally:
        session.close()
        
        
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

    
    
####### Hilfsfunktionen für Create Output
def get_its():
    DATABASE_URL = "sqlite:///database.db"
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    
    session = Session()
    try:
        entries = session.query(ITSBaseData).all()
        data = [entry.__dict__ for entry in entries]
        df = pd.DataFrame(data).drop(columns=["_sa_instance_state"], errors='ignore')
        #print(df.head(2))
        
        # Zuerst nach table_id sortieren
        df_sorted = df.sort_values(by='table_id', ascending=True)
        
        # Dann Duplikate in konzept_code entfernen (nach Sortierung, also kleinste table_id bleibt)
        df_unique = df_sorted.drop_duplicates(subset='konzept_code', keep='first')
        
        # Ausgabe der eindeutigen table_ids, sortiert (basierend auf dem bereinigten DataFrame)
        table_ids_sorted = sorted(df_unique['table_id'].dropna().unique())
        print("Eindeutige table_ids (aufsteigend sortiert):")
        print(table_ids_sorted)

        return df_unique[['konzept_code', 'y_axis_rc_code', 'table_id', 'y_axis_name', 'x_axis_name', 'x_axis_rc_code']]
    finally:
        session.close()


def find_node_by_componentcode(node, componentcode):
    if node.componentcode == componentcode:
        return node
    for child in node.children:
        result = find_node_by_componentcode(child, componentcode)
        if result:
            return result
    return None

def find_node_in_tree(roots, componentcode):
    for root in roots:
        result = find_node_by_componentcode(root, componentcode)
        if result:
            return result
    return None



def create_output():
    print("hello world8")

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




