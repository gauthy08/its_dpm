Zielprojektstruktur

project/
├── data/
│   └── START_Konzepte.csv
├── database/
│   └── models.py         # Datenbankmodelle
│   └── db_manager.py     # DB-Interaktionen (Session, Tabellen-Handling)
├── scripts/
│   └── load_data.py      # CSV-Daten in DB laden
    └── merge_data.py      # CSV-Daten mergen
└── main.py               # Hauptprogramm



Start:
über Terminal mit "python main.py"

To dos: 
START_Konzepte.db erstellen: check
ITSBaseData erstellen: check 
implement pandasDF[pandasDF["taxonomy_code"] == "FINREP_3.2.1"]: check
Import DPM data: check
Import reference data : check
START Konzepte mit ITS Base Data und DPM mergen: check

open:
merge reference mit merge Y und X ? 
