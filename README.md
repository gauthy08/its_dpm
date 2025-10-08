## ITS datapoint description/data catalogue 

# Beschreibung
Mit diesem Projekt wird mit Hilfe des OeNB Chatbots Beschreibungen für ITS Datenpunkte generiert, welche im Anschluss an ISIS und infolgedessen an den Datenkatalog hochgeladen werden. 

# Voraussetzung
Mit den folgenden Einstellungen sollte das Projekt ausführbar sein
Python 3.12
Enable Spark
Spark 3.3.2 - CDP 7.1.9.0 

# Benutzung
Start:
im Terminal "python main.py" aufrufen. Dann wird ein "Menü" angezeicht mit verfügbaren Funktionen. 

## 1: Tabellen erstellen/create_tables()
Das Projekt beinhaltet eine eigene kleine Datenbank databank.db mit sqlalchemy. Falls man diese komplett löscht, können die Tabellen neu erstellt werden. Ist aber im Standardfall nicht notwendig. Kann man also grundsätzlich ignorieren, sondern muss/kann nur ausgeführt werden, wenn man bei "Null" beginnt. 
Noch offenes TO DO: entferne Tabellen nicht mehr gebraucht werden. 

## 2: ITS Base Data laden/load_hue_its()
Diese Funktion lädt ITS-Basisdaten aus der Hive-Datenbank its_analysedaten_prod über eine SparkSession mit HiveWarehouseConnector. Die Daten werden direkt bei der Abfrage auf die zwei relevanten Taxonomy-Codes FINREP_3.2.1 und COREP_3.2 gefiltert, um die Datenmenge zu reduzieren (ca. 81.000 Zeilen). Nach der Abfrage werden die Spark-DataFrames in Pandas DataFrames konvertiert und anschließend in die lokale SQL-Datenbank-Tabelle ITSBaseData_new geschrieben. Die Funktion bietet dabei interaktive Optionen zum Umgang mit bereits vorhandenen Daten (löschen, hinzufügen oder abbrechen) und zeigt während des gesamten Prozesses Fortschrittsmeldungen an. Ein NumPy-Kompatibilitäts-Workaround sorgt für die reibungslose Konvertierung zwischen älteren PySpark- und neueren NumPy-Versionen.
Muss nicht immer ausgeführt werden, da die Daten ja in der Projekt-Datenbank erhalten bleiben. Muss nur durchgeführt werden, wenn beispielsweise ein neues Framework (Finrep oder Corep) verwendet wird. 

## 3: DPM_TableStructure hochladen/Hierachie
`load_tablestructurehierarchy(file_path)` konvertiert flache Excel-Tabellenstrukturen (qDPM_TableStructure) aus dem Ordner "data" in hierarchische Baumstrukturen für z.B. COREP 3.2 Regulatory Reporting. Erstellt für jeden TableCode zwei separate Bäume - einen für Zeilen ("Table row") und einen für Spalten ("Table column") - basierend auf den Parent-Child-Beziehungen über OrdinateID/ParentOrdinateID. Serialisiert alle generierten Baumstrukturen als Dictionary in eine Pickle-Datei (baumstruktur_XXX.pkl) im Ordner tree_structures für spätere Verwendung. Diese Baumstrukturen werden genutzt, um hierarchische Kontexte für die RAG-basierte Textgenerierung zu erstellen - jeder Datenpunkt kann so mit seinem vollständigen hierarchischen Pfad beschrieben werden.
Siehe "README_TREE_VIEWER.md": Damit kann man die Baumstruktur visualisieren/analysieren zum besseren Verständnis. 

## 4: PRODUCTION: ChatBot-Beschreibungen generieren 

### Zweck
Generiert automatisch KI-basierte Erklärungen für regulatorische Datenpunkte aus COREP/FINREP-Templates mithilfe eines LLM (Mistral).
Was passiert:
	1. Lädt Baumstruktur aus Pickle-File (z.B. alle COREP-Tabellen mit hierarchischen Beziehungen)
	2. Durchläuft jeden Datenpunkt im Baum (z.B. "0010: Common Equity Tier 1 capital")
	3. Erstellt Kontext mit dem hierarchischen Pfad (Parent → Child Beziehungen)
	4. Sendet an KI-API mit RAG-Knowledge-Base für regulatorisches Wissen
	5. Erhält Beschreibung zurück (max. 100 Wörter, faktisch, ohne Halluzinationen)
	6. Speichert Ergebnisse als Excel in production_runs/
### Input
	• Pickle-File mit Baumstrukturen
	• Prompt-Template (definiert Antwortformat)
	• Scope (welche Tabellen verarbeiten)
### Output
Excel-File mit:
	• Alle Datenpunkte + ihre generierten Beschreibungen
	• Hierarchischer Kontext pro Datenpunkt
	• Metadaten (Verarbeitungszeit, verwendete Parameter)
Beispiel-Ergebnis:
Node: "0010 - Common Equity Tier 1 capital"
Generated: "Common Equity Tier 1 capital refers to the highest quality 
           regulatory capital that banks must hold, consisting primarily 
           of common shares and retained earnings..."

offene TODO: Menüpunkt-Auswahl steckt aktuell in in main.py. Könnte man zur besseren Übersicht in eine eigene Funktion stecken. 
Output-pfad/files beschreiben anpassen. 


## 5: Corep Annex 5 extrahieren


## 4: 🚀 PRODUCTION: ChatBot-Beschreibungen generieren


???Menüpunkt 1: Die Funktion `create_tables()` erstellt automatisch alle in den SQLAlchemy-Modellen definierten Tabellen, **sofern sie noch nicht existieren**. Bereits bestehende Tabellen bleiben **unverändert**. 

???Menüpunkt 2: `load_hue_its()` ist eine ETL-Funktion, die Regulatory Reporting Daten aus der HUE-Datenbank its_analysedaten_wapr.its_base_data über Spark/Hive Warehouse Connector lädt. Filtert automatisch nach FINREP 3.2.1 und COREP 3.2 Taxonomien und bietet interaktive Optionen für den Umgang mit bereits vorhandenen Daten (löschen/erweitern/abbrechen). Schreibt die gefilterten Daten als Batch-Insert in die lokale SQL-Datenbank-Tabelle ITSBaseData_new.

???Menüpunkt 3: `load_tablestructurehierarchy(file_path)` konvertiert flache Excel-Tabellenstrukturen (qDPM_TableStructure) in hierarchische Baumstrukturen für z.B. COREP 3.2 Regulatory Reporting. Erstellt für jeden TableCode zwei separate Bäume - einen für Zeilen ("Table row") und einen für Spalten ("Table column") - basierend auf den Parent-Child-Beziehungen über OrdinateID/ParentOrdinateID. Serialisiert alle generierten Baumstrukturen als Dictionary in eine Pickle-Datei (baumstruktur_XXX.pkl) im Ordner tree_structures für spätere Verwendung. TO DO: Corep oder Finrep soll nicht in der load-funktion sondern im Menü ausgewählt werden. 


# gitignore 


# Refactoring: 
to do: create_tables() nicht benötigte Tables entfernen bzw. löschen