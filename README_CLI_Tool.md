# Database CLI Tool

Einfaches Command-Line Interface für ITS Base Data Abfragen. Ersetzt Jupyter Notebooks.

## 🚀 Setup

```bash
pip install click pandas sqlalchemy
python db_cli.py --help
```

## 📋 Kommandos

### Verfügbare Tabellen
```bash
python db_cli.py tables
```

### Daten anzeigen
```bash
python db_cli.py list its --limit 10
python db_cli.py info its
```

### Konzept Code suchen (Hauptfunktion)
```bash
# Direkter Lookup
python db_cli.py konzept ISFIN0000001

# Feld-spezifische Suche
python db_cli.py search its "ISFIN0000001" --field ko
```

### Allgemeine Suche
```bash
python db_cli.py search its "Balance Sheet"
python db_cli.py search its "F 01.01" --field template_code
python db_cli.py search its "FINREP" --field taxonomy_code
```

### Multi-Filter Suche (NEU!)
```bash
# Kombiniere mehrere Filter
python db_cli.py filter its --taxonomy FINREP --template "F 01.01"
python db_cli.py filter its --ko ISFIN --module FINREP9
python db_cli.py filter its --taxonomy COREP --datapoint 112718
python db_cli.py filter its --template "Balance" --module FINREP9 --export results.csv
```

### Verfügbare Suchfelder anzeigen
```bash
python db_cli.py fields its
```

### Multi-Filter Suche (kombiniere Kriterien)
```bash
python db_cli.py filter its --taxonomy FINREP --template "F 01.01"
python db_cli.py filter its --ko ISFIN --module FINREP9 --export results.csv
```

### Export
```bash
python db_cli.py export its data.csv
python db_cli.py export its data.csv --limit 1000
```

### SQL-Abfragen
```bash
python db_cli.py query its "SELECT COUNT(*) FROM ITSBaseData_new"
python db_cli.py query its "SELECT * FROM ITSBaseData_new WHERE ko='ISFIN0000001'" --output results.csv
```

## 🎯 Häufige Workflows

```bash
# 1. Konzept analysieren
python db_cli.py konzept ISFIN0000001
python db_cli.py query its "SELECT * FROM ITSBaseData_new WHERE ko='ISFIN0000001'" --output konzept_details.csv

# 2. Template-Daten finden
python db_cli.py search its "F 01.01" --field template_code
python db_cli.py filter its --template "F 01.01" --taxonomy FINREP --export balance_sheet.csv

# 3. Spezifische Kombination finden
python db_cli.py filter its --taxonomy FINREP --module FINREP9 --limit 20
python db_cli.py filter its --ko ISFIN --template "Balance" --export isfin_balance.csv
```

## 📊 Wichtige Suchfelder

| Feld | Beschreibung | Beispiel |
|------|-------------|----------|
| `ko` | Konzept Code | `ISFIN0000001` |
| `datapoint` | Datapoint ID | `112718` |
| `taxonomy_code` | Taxonomy | `FINREP_3.2.1` |
| `template_code` | Template | `F 01.01` |
| `module_code` | Module | `FINREP9` |

## 🔧 Tipps

- Nutzen Sie `--limit` bei großen Suchen
- `python db_cli.py konzept <code>` für direkten Konzept-Lookup
- `python db_cli.py filter its --taxonomy X --template Y` für präzise Multi-Filter
- `python db_cli.py fields its` zeigt alle durchsuchbaren Felder
- Alle Kommandos haben `--help` Option

---

**Hauptfunktion**: Konzept Code Suche in ITS Base Data