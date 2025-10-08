import pandas as pd
import re
from datetime import datetime
from sqlalchemy.orm import Session
from pyspark.sql import SparkSession
from pyspark_llap import HiveWarehouseSession
import pickle
from .tree import Node

# Lokale Module
from database.db_manager import SessionLocal, create_tables
from database.models import Template_Finrep, Finrep_Y_reference, DPM_tableStructure, ITSBaseData_new

def load_hue_its():
    import numpy as np
    # Workaround für NumPy Kompatibilitätsproblem
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float_
    np.complex = np.complex_
    np.object = np.object_
    np.str = np.str_
    
    hue_database = "its_analysedaten_prod"
    table_name = "its_base_data"
    
    print(f"📊 Starte Datenabfrage aus {hue_database}.{table_name}")
    print("=" * 60)
    
    # Spark-Session initialisieren
    spark = SparkSession.builder\
        .appName("hwc-app")\
        .config("spark.security.credentials.hiveserver2.enabled", "false")\
        .config("spark.datasource.hive.warehouse.read.via.llap", "false")\
        .config("spark.datasource.hive.warehouse.read.jdbc.mode", "client")\
        .config("spark.sql.hive.hiveserver2.jdbc.url",
                "jdbc:hive2://anucdp-mgmt-01.w.oenb.co.at:2181,anucdp-mgmt-02.w.oenb.co.at:2181,anucdp-mgmt-03.w.oenb.co.at:2181/;"
                "serviceDiscoveryMode=zooKeeper;zooKeeperNamespace=hiveserver2;trustStoreType=jks;ssl=true")\
        .config("spark.yarn.historyServer.address", "https://anucdp-mgmt-02.w.oenb.co.at")\
        .config("spark.sql.hive.hiveserver2.jdbc.url.principal", "hive/_HOST@AD.OENB.CO.AT")\
        .config("spark.hadoop.yarn.resourcemanager.principal", "hive")\
        .config("spark.kryo.registrator", "com.qubole.spark.hiveacid.util.HiveAcidKyroRegistrator")\
        .config("spark.sql.extensions", "com.qubole.spark.hiveacid.HiveAcidAutoConvertExtension")\
        .config("spark.jars",
                "/runtime-addons/spark332-7190-1202-b75-ht9wmb/opt/spark/optional-lib/hive-warehouse-connector-assembly.jar")\
        .getOrCreate()

    print("✅ Spark Session erstellt")

    try:
        # HiveWarehouseSession initialisieren
        hwc = HiveWarehouseSession.session(spark).build()
        print("✅ HiveWarehouseSession initialisiert")

        # Direkt gefilterte Daten abfragen - NUR die zwei gewünschten Taxonomy Codes
        sql_query = f"""
        SELECT * 
        FROM {hue_database}.{table_name}
        WHERE taxonomy_code IN ('FINREP_3.2.1', 'COREP_3.2')
        """
        
        print(f"📝 Führe gefilterte Query aus (nur FINREP_3.2.1 und COREP_3.2)")
        
        # DataFrame erstellen
        filtered_df = hwc.sql(sql_query)
        
        # Anzahl der Zeilen prüfen
        row_count = filtered_df.count()
        print(f"📊 Gefundene Zeilen: {row_count}")
        
        # Alternative Methode: Daten mit collect() sammeln und dann zu Pandas konvertieren
        print("🔄 Sammle Daten...")
        data = filtered_df.collect()
        
        print("🔄 Konvertiere zu Pandas DataFrame...")
        # Manuelle Konvertierung zu Pandas DataFrame
        pandas_df = pd.DataFrame([row.asDict() for row in data])
        
        print(f"✅ Daten erfolgreich geladen: {len(pandas_df)} Zeilen")
        
        # Zur Kontrolle: Welche Taxonomy Codes sind tatsächlich vorhanden
        unique_taxonomies = pandas_df["taxonomy_code"].unique().tolist()
        print(f"📋 Geladene Taxonomy Codes: {unique_taxonomies}")

        # Verbindung zur Datenbank
        session = SessionLocal()
        try:
            count_existing = session.query(ITSBaseData_new).count()
            if count_existing > 0:
                print(f"⚠️  Achtung: Es sind bereits {count_existing} Einträge in der Tabelle vorhanden.")
                user_input = input("Möchtest du die Tabelle löschen (l), nur neue Daten einfügen (n), oder den Vorgang abbrechen (a)? [l/n/a]: ").lower()

                if user_input == "a":
                    print("❌ Vorgang abgebrochen.")
                    return

                elif user_input == "l":
                    print("🔄 Lösche vorhandene Daten...")
                    session.query(ITSBaseData_new).delete()
                    session.commit()

                elif user_input != "n":
                    print("❌ Ungültige Eingabe – Vorgang wird abgebrochen.")
                    return

            # Daten einfügen
            print(f"💾 Füge {len(pandas_df)} Zeilen in die Datenbank ein...")
            
            for idx, row in pandas_df.iterrows():
                if idx % 1000 == 0:  # Progress-Anzeige alle 1000 Zeilen
                    print(f"   Verarbeite Zeile {idx}/{len(pandas_df)}...")
                    
                its_data = ITSBaseData_new(
                    datapoint=row['datapoint'],
                    ko=row['ko'],
                    taxonomy_code=row['taxonomy_code'],
                    template_code=row['template_code'],
                    template_label=row['template_label'],
                    module_code=row['module_code'],
                    module_gueltig_von=row['module_gueltig_von'] if pd.notnull(row['module_gueltig_von']) else None,
                    module_gueltig_bis=row['module_gueltig_bis'] if pd.notnull(row['module_gueltig_bis']) else None,
                    table_code=row['table_code'],
                    table_name=row['table_name'],
                    criteria=row['criteria'],
                    x_axis_rc_code=row['x_axis_rc_code'],
                    x_axis_name=row['x_axis_name'],
                    y_axis_rc_code=row['y_axis_rc_code'],
                    y_axis_name=row['y_axis_name'],
                    z_axis_rc_code=row['z_axis_rc_code'],
                    z_axis_name=row['z_axis_name']
                )
                session.add(its_data)

            session.commit()
            print("✅ Daten erfolgreich in die Datenbank geschrieben.")
            
        except Exception as e:
            print(f"❌ Fehler beim Schreiben in die Datenbank: {e}")
            session.rollback()
        finally:
            session.close()
            
    except Exception as e:
        print(f"❌ Fehler bei der Datenabfrage: {e}")
    finally:
        # Spark Session schließen
        spark.stop()
        print("🔒 Spark Session geschlossen")


        
        
# Funktion zum Laden der Daten aus der CSV-Datei und Schreiben in die Datenbank
def load_dpm_to_db(file_path):
    """
    Lädt Daten aus einer CSV-Datei und schreibt sie in die Datenbank.
    """
    # CSV-Daten laden
    #df = pd.read_csv(file_path, delimiter=';', encoding='Windows-1252', header=None, names=["DataPointVID", "DataPointID", "DimensionLabel", "MemberName"])
    
    df = pd.read_csv(file_path, delimiter=';', encoding='Windows-1252', header=None, names= ["DataPointVID", "DataPointID", "DimensionLabel", "MemberName"])

    
    print(df.head())

    # Verbindung zur Datenbank herstellen
    session = SessionLocal()

    try:
        # DataFrame-Zeilen iterieren und in die Datenbank einfügen
        for _, row in df.iterrows():
            datapoint = DPM_datapoint(
                datapoint_vid=row['DataPointVID'],
                datapoint_id=row['DataPointID'],
                dimension_label=row['DimensionLabel'],
                member_name=row['MemberName'],
            )
            session.add(datapoint)
        
        # Änderungen speichern
        session.commit()
        print("Daten erfolgreich in die Datenbank geladen.")
    except Exception as e:
        print(f"Fehler: {e}")
        session.rollback()
    finally:
        session.close()
 


def read_excel_data(file_path):
    """
    Liest Daten aus einer Excel-Datei ein und schreibt sie in die Datenbank.
    
    Erwartet wird eine Excel-Datei mit den Spalten:
    "Worksheet", "Table", "Axis", "Coord", "Text", "Reference", "Extra".
    """
    # Excel-Daten laden und 'Coord' als String einlesen
    try:
        df = pd.read_excel(file_path, dtype={'Coord': str})
    except Exception as e:
        print(f"Fehler beim Laden der Excel-Datei: {e}")
        return

    print(df.head())

    # Verbindung zur Datenbank herstellen (SessionLocal muss vorher definiert worden sein)
    session = SessionLocal()

    try:
        # Iteriere über die Zeilen des DataFrames und füge sie der Datenbank hinzu
        for _, row in df.iterrows():
            try:
                template_sheet = int(row['Worksheet'])
            except (ValueError, TypeError):
                template_sheet = None  # oder einen Default-Wert setzen

            record = Template_Finrep(
                template_sheet = template_sheet,
                table          = row['Table'],
                axis           = row['Axis'],
                coord          = row['Coord'],  # Dieser Wert bleibt nun z. B. "0010"
                text           = row['Text'],
                reference      = row['Reference'],
                extra          = row['Extra']
            )
            session.add(record)
        
        # Änderungen speichern
        session.commit()
        print("Daten erfolgreich in die Datenbank geladen.")
    except Exception as e:
        print(f"Fehler: {e}")
        session.rollback()
    finally:
        session.close()

def load_finrep_y_reference(file_path):
    print("ddd")
    
    # 1) Excel-Daten laden
    #df = pd.read_excel(file_path, sheet_name=0, dtype={'Coord': str})  
    df = pd.read_csv(file_path, dtype={'Coord': str})  
    
    print(df.head())
    
    # 2) Verbindung zur Datenbank
    session = SessionLocal()
    
    try:
        # 3) DataFrame-Zeilen iterieren und in die Datenbank einfügen
        for _, row in df.iterrows():
            merged = Finrep_Y_reference(
                worksheet=row.get('Worksheet'),
                table_=row.get('Table'),
                axis=row.get('Axis'),
                coord=row.get('Coord'),
                text=row.get('Text'),
                reference=row.get('Reference'),
                extra=row.get('Extra'),
                rag_text=row.get('RAG_Text'),
            )
            session.add(merged)

        # 4) Änderungen speichern
        session.commit()
    
    except Exception as e:
        print(f"Fehler beim Import: {e}")
        session.rollback()
    
    finally:
        session.close()

        

import os
import pandas as pd
import pickle

def load_tablestructurehierarchy(file_path, taxonomy_code="COREP 3.2"):
    """
    Liest die qDPM_TableStructure aus der Excel-Datei ein und erzeugt 
    für jeden TableCode (filtert nach gewähltem TaxonomyCode) 
    jeweils zwei Bäume: 
      - Einen für ComponentTypeName = 'Table row'
      - Einen für ComponentTypeName = 'Table column'
    Speichert alle Bäume in einem Dictionary und 
    serialisiert dieses Dictionary mit Pickle im Ordner "tree_structures".
    
    Args:
        file_path (str): Pfad zur Excel-Datei
        taxonomy_code (str): TaxonomyCode zum Filtern (z.B. "FINREP 3.2.1", "COREP 3.2")
    """
    print(f"Lade Datei: {file_path}")
    print(f"Verwende TaxonomyCode: {taxonomy_code}")
    
    # 1) Vollständige Tabelle laden
    df = pd.read_excel(file_path, sheet_name="qDPM_TableStructure",
                       dtype={'ComponentCode': str, 'Level': str, 'x_axis_rc_code': str, 'y_axis_rc_code': str})
    
    # 2) Nach gewähltem TaxonomyCode filtern
    df = df[df['TaxonomyCode'] == taxonomy_code]
    
    # 3) Alle relevanten TableCodes ermitteln
    all_table_codes = df['TableCode'].unique()
    
    # 4) Dictionary, in dem später alle Bäume gespeichert werden:
    #    Key: (table_code, component_type), Value: Liste von Wurzelknoten (roots)
    all_trees = {}
    
    # 5) Für jeden TableCode jeweils 'Table row' UND 'Table column' verarbeiten
    for table_code in all_table_codes:
        for comp_type in ['Table row', 'Table column']:
            # Teil-DataFrame für diese Kombination
            subdf = df[
                (df['TableCode'] == table_code) &
                (df['ComponentTypeName'] == comp_type)
            ]
            
            # Falls es dazu keine Zeilen gibt (z.B. kein "Table column" vorhanden),
            # dann überspringen wir diese Kombination
            if subdf.empty:
                continue
            
            print(f"Erzeuge Baum für TableCode='{table_code}', ComponentType='{comp_type}'")
            
            # 5a) Knoten anlegen (Dictionary: {OrdinateID: Node})
            nodes = {}
            for _, row in subdf.iterrows():
                parent_id = row["ParentOrdinateID"] if pd.notnull(row["ParentOrdinateID"]) else None
                
                node = Node(
                    tablecode           = row["TableCode"],
                    componenttypename   = row["ComponentTypeName"],
                    componentcode       = row["ComponentCode"],
                    componentlabel      = row["ComponentLabel"],
                    headerflag          = row["HeaderFlag"],
                    level               = row["Level"],
                    order               = row["Order"],
                    taxonomycode        = row["TaxonomyCode"],
                    displaybeforechildren = row["DisplayBeforeChildren"],
                    ordinateid          = row["OrdinateID"],
                    parentordinateide   = parent_id
                )
                nodes[row["OrdinateID"]] = node
                
            # 5b) Eltern-Kind-Beziehungen setzen
            for node in nodes.values():
                parent_id = node.parentordinateide
                if parent_id in nodes:
                    node.parent = nodes[parent_id]
                    nodes[parent_id].children.append(node)
                    
            # 5c) Wurzelknoten ermitteln (Nodes ohne Parent)
            roots = [n for n in nodes.values() if n.parent is None]
            
            # 5d) Im Dictionary speichern
            all_trees[(table_code, comp_type)] = roots
    
    # 6) Optional: Beispielausgabe
    #    Man könnte zum Testen z.B. mal alle Keys (TableCode, ComponentType) ausgeben
    print(f"\nErzeugte Bäume für folgende (TableCode, ComponentType)-Kombinationen:")
    for k in all_trees.keys():
        print("  ", k)
    
    # 7) Ordner "tree_structures" erstellen, falls er nicht existiert
    tree_structures_dir = "tree_structures"
    os.makedirs(tree_structures_dir, exist_ok=True)
    
    # 8) Dateiname basierend auf TaxonomyCode generieren
    # Leerzeichen und Punkte durch Unterstriche ersetzen für gültigen Dateinamen
    safe_taxonomy_name = taxonomy_code.replace(" ", "_").replace(".", "_")
    pickle_filename = f"baumstruktur_{safe_taxonomy_name}.pkl"
    pickle_path = os.path.join(tree_structures_dir, pickle_filename)
    with open(pickle_path, "wb") as f:
        pickle.dump(all_trees, f)
    
    print(f"\nAlle Bäume wurden in '{pickle_path}' gespeichert.")
    return all_trees
        
def extract_corep_annex():
    print("COREP EXTRACT")