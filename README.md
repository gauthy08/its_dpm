## ITS datapoint description/data catalogue 

# Beschreibung
Mit diesem Projekt werden mit Hilfe von LLMs bzw. des OeNB Chatbots Beschreibungen für ITS Datenpunkte generiert, welche im Anschluss an ISIS und infolgedessen an den Datenkatalog hochgeladen werden. 

# Voraussetzung
Session mit "enable Spark" starten

# Benutzung
Start:
im Terminal "python main.py" aufrufen 

Menüpunkt 1: Die Funktion `create_tables()` erstellt automatisch alle in den SQLAlchemy-Modellen definierten Tabellen, **sofern sie noch nicht existieren**. Bereits bestehende Tabellen bleiben **unverändert**. 

Menüpunkt 2: `load_hue_its()` ist eine ETL-Funktion, die Regulatory Reporting Daten aus der HUE-Datenbank its_analysedaten_wapr.its_base_data über Spark/Hive Warehouse Connector lädt. Filtert automatisch nach FINREP 3.2.1 und COREP 3.2 Taxonomien und bietet interaktive Optionen für den Umgang mit bereits vorhandenen Daten (löschen/erweitern/abbrechen). Schreibt die gefilterten Daten als Batch-Insert in die lokale SQL-Datenbank-Tabelle ITSBaseData_new.

Menüpunkt 3: `load_tablestructurehierarchy(file_path)` konvertiert flache Excel-Tabellenstrukturen (qDPM_TableStructure) in hierarchische Baumstrukturen für z.B. COREP 3.2 Regulatory Reporting. Erstellt für jeden TableCode zwei separate Bäume - einen für Zeilen ("Table row") und einen für Spalten ("Table column") - basierend auf den Parent-Child-Beziehungen über OrdinateID/ParentOrdinateID. Serialisiert alle generierten Baumstrukturen als Dictionary in eine Pickle-Datei (baumstruktur_XXX.pkl) im Ordner tree_structures für spätere Verwendung. TO DO: Corep oder Finrep soll nicht in der load-funktion sondern im Menü ausgewählt werden. 


# Refactoring: 
choice == "2": load_csv_to_db("data/ISIS-Erhebungsstammdaten1.xlsx") wird/wurde entfernt

load_dpm_to_db("data/qDPM_DataPointCategorisations.csv") wird das noch gebraucht? Offen! 

aktuelles to do: entfern von 
choice == "6":
        merge_data()
        update_merged_data_with_dpm()