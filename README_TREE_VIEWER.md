# COREP Tree Viewer

Ein interaktives Konsolen-Tool zur Visualisierung und Analyse von COREP/FINREP Tabellenstrukturen.

## Überblick

Der Tree Viewer lädt hierarchische Baumstrukturen aus Pickle-Dateien und bietet eine benutzerfreundliche Konsolenanwendung zur Navigation und Analyse der Daten.

## Funktionen

### 🌳 **Baumvisualisierung**
- Anzeige kompletter Hierarchien mit Einrückungen
- Übersicht über Wurzelknoten und Kindstrukturen
- Begrenzte Tiefendarstellung für bessere Lesbarkeit

### 📊 **Statistiken**
- Knotenzählung (gesamt, Blattknoten, Wurzeln)
- Maximale Baumtiefe
- Übersicht für einzelne Bäume oder alle Strukturen

### 🔍 **Suche**
- Durchsuchen von Knoten nach Code oder Label
- Pfadanzeige zu gefundenen Elementen
- Case-insensitive Suche

### 💾 **Export**
- JSON-Export einzelner Bäume oder aller Strukturen
- UTF-8 Kodierung mit lesbarer Formatierung

## Verwendung

```bash
python tree_viewer.py
```

Das Tool lädt automatisch Pickle-Dateien aus dem `tree_structures/` Ordner und startet das interaktive Menü.

## Dateistruktur

```
tree_structures/
├── baumstruktur_COREP_3_2.pkl
├── baumstruktur_FINREP_3_2_1.pkl
└── ...
```

## Systemanforderungen

- Python 3.x
- Module: `pickle`, `os`, `json`
- Konsole mit Clearing-Unterstützung