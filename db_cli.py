#!/usr/bin/env python3
"""
Simple Database CLI Tool
Einfaches, erweiterbares CLI für Datenbankabfragen

Usage:
    python db_cli.py list its --limit 10
    python db_cli.py search its "Liquidity"
    python db_cli.py export its results.csv
"""

import click
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import ITSBaseData_new
from pathlib import Path

# Database setup
DATABASE_URL = "sqlite:///database.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

class TableManager:
    """Basis Manager für Tabellen-Operationen"""
    
    def __init__(self, model_class):
        self.model_class = model_class
        self.session = SessionLocal()
    
    def close(self):
        self.session.close()
    
    def to_dataframe(self, entries):
        """SQLAlchemy Entries zu DataFrame"""
        data = []
        for entry in entries:
            row = entry.__dict__.copy()
            row.pop("_sa_instance_state", None)
            data.append(row)
        return pd.DataFrame(data)
    
    def get_all(self, limit=None):
        """Alle Einträge holen"""
        query = self.session.query(self.model_class)
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def count(self):
        """Anzahl Einträge"""
        return self.session.query(self.model_class).count()

# Verfügbare Tabellen - hier einfach erweiterbar
TABLES = {
    'its': {
        'model': ITSBaseData_new,
        'description': 'ITS Base Data'
    }
    # Später hinzufügen:
    # 'konzept': {
    #     'model': Konzept,
    #     'description': 'Konzept Data'
    # },
    # 'merged': {
    #     'model': MergedData,
    #     'description': 'Merged Data'
    # }
}

def get_table_manager(table_name):
    """Table Manager für gegebene Tabelle erstellen"""
    if table_name not in TABLES:
        available = ', '.join(TABLES.keys())
        raise click.ClickException(f"Unbekannte Tabelle '{table_name}'. Verfügbar: {available}")
    
    return TableManager(TABLES[table_name]['model'])

@click.group()
def cli():
    """🗄️ Simple Database CLI Tool"""
    pass

@cli.command()
def tables():
    """📋 Zeige verfügbare Tabellen"""
    click.echo("Verfügbare Tabellen:")
    for name, info in TABLES.items():
        click.echo(f"  {name:<10} - {info['description']}")

@cli.command()
@click.argument('table_name')
@click.option('--limit', '-l', default=20, help='Anzahl Zeilen (default: 20)')
def list(table_name, limit):
    """📄 Liste Einträge einer Tabelle"""
    
    manager = get_table_manager(table_name)
    try:
        entries = manager.get_all(limit)
        
        if not entries:
            click.echo("❌ Keine Einträge gefunden")
            return
        
        df = manager.to_dataframe(entries)
        
        # Zeige nur wichtige Spalten für bessere Übersicht
        if table_name == 'its':
            key_cols = ['id', 'taxonomy_code', 'template_code', 'table_name']
        else:
            key_cols = df.columns[:4].tolist()  # Erste 4 Spalten
        
        available_cols = [col for col in key_cols if col in df.columns]
        display_df = df[available_cols] if available_cols else df
        
        click.echo(f"📄 {table_name.upper()} - Erste {len(entries)} Einträge:")
        click.echo("=" * 60)
        click.echo(display_df.to_string(index=False))
        
        total = manager.count()
        click.echo(f"\nGesamt in Tabelle: {total:,} Einträge")
        
    finally:
        manager.close()

@cli.command()
@click.argument('table_name')
@click.argument('search_term')
@click.option('--limit', '-l', default=50, help='Max Suchergebnisse')
@click.option('--field', help='Spezifische Spalte durchsuchen (z.B. taxonomy_code, datapoint)')
def search(table_name, search_term, limit, field):
    """🔍 Suche in Tabelle"""
    
    manager = get_table_manager(table_name)
    try:
        if table_name == 'its':
            query = manager.session.query(ITSBaseData_new)
            
            if field:
                # Suche in spezifischer Spalte
                if hasattr(ITSBaseData_new, field):
                    column = getattr(ITSBaseData_new, field)
                    if field == 'datapoint':
                        # Für numerische Suche
                        try:
                            query = query.filter(column == int(search_term))
                        except ValueError:
                            click.echo(f"❌ '{search_term}' ist keine gültige Nummer für {field}")
                            return
                    else:
                        # Für Text-Suche
                        query = query.filter(column.contains(search_term))
                else:
                    click.echo(f"❌ Spalte '{field}' existiert nicht in {table_name}")
                    return
            else:
                # Standard-Suche in wichtigen Feldern
                query = query.filter(
                    (ITSBaseData_new.template_label.contains(search_term)) |
                    (ITSBaseData_new.table_name.contains(search_term)) |
                    (ITSBaseData_new.taxonomy_code.contains(search_term)) |
                    (ITSBaseData_new.template_code.contains(search_term)) |
                    (ITSBaseData_new.module_code.contains(search_term))
                )
            
            entries = query.limit(limit).all()
        else:
            # Fallback für andere Tabellen
            entries = manager.get_all(limit)
        
        if not entries:
            field_info = f" in Spalte '{field}'" if field else ""
            click.echo(f"❌ Keine Ergebnisse für '{search_term}'{field_info}")
            return
        
        df = manager.to_dataframe(entries)
        
        # Zeige relevante Spalten
        if table_name == 'its':
            cols = ['id', 'datapoint', 'taxonomy_code', 'template_code', 'table_name']
        else:
            cols = df.columns[:4].tolist()
        
        available_cols = [col for col in cols if col in df.columns]
        display_df = df[available_cols] if available_cols else df
        
        field_info = f" (Feld: {field})" if field else ""
        click.echo(f"🔍 Suchergebnisse '{search_term}'{field_info} in {table_name.upper()}: {len(entries)} gefunden")
        click.echo("=" * 90)
        click.echo(display_df.to_string(index=False))
        
    finally:
        manager.close()

@cli.command()
@click.argument('konzept_code')
@click.option('--table', default='its', help='Tabelle durchsuchen (default: its)')
def konzept(konzept_code, table):
    """🎯 Suche nach Konzept Code (in ko Spalte)"""
    
    manager = get_table_manager(table)
    try:
        if table == 'its':
            # Konzept Code steht in der 'ko' Spalte
            entries = manager.session.query(ITSBaseData_new).filter(
                ITSBaseData_new.ko == konzept_code
            ).all()
        else:
            # Für andere Tabellen später
            entries = []
        
        if not entries:
            click.echo(f"❌ Konzept Code '{konzept_code}' nicht gefunden")
            click.echo("💡 Tipp: Versuchen Sie 'python db_cli.py search its <code>' für allgemeine Suche")
            return
        
        df = manager.to_dataframe(entries)
        
        click.echo(f"🎯 Konzept Code '{konzept_code}': {len(entries)} Treffer gefunden")
        click.echo("=" * 90)
        
        # Zeige alle relevanten Spalten inklusive ko
        if table == 'its':
            cols = ['id', 'datapoint', 'ko', 'taxonomy_code', 'template_code', 'table_name']
        else:
            cols = df.columns[:6].tolist()
        
        available_cols = [col for col in cols if col in df.columns]
        display_df = df[available_cols] if available_cols else df
        click.echo(display_df.to_string(index=False))
        
        # Zusätzliche Details
        if len(entries) == 1:
            entry = entries[0]
            click.echo(f"\n📋 Details:")
            click.echo(f"  Datapoint:  {entry.datapoint}")
            click.echo(f"  Template:   {entry.template_code}")
            click.echo(f"  Module:     {entry.module_code}")
            click.echo(f"  Taxonomy:   {entry.taxonomy_code}")
        
    finally:
        manager.close()

@cli.command()
@click.argument('table_name')
@click.argument('output_file')
@click.option('--limit', '-l', default=0, help='Limit (0 = alle)')
def export(table_name, output_file, limit):
    """💾 Exportiere Tabelle zu CSV"""
    
    manager = get_table_manager(table_name)
    try:
        entries = manager.get_all(limit if limit > 0 else None)
        
        if not entries:
            click.echo("❌ Keine Daten zum Exportieren")
            return
        
        df = manager.to_dataframe(entries)
        
        # Erstelle Output-Verzeichnis falls nötig
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        click.echo(f"✅ {len(df)} Einträge aus {table_name.upper()} exportiert nach {output_file}")
        click.echo(f"📊 {len(df.columns)} Spalten")
        
    finally:
        manager.close()

@cli.command()
@click.argument('table_name')
def info(table_name):
    """ℹ️ Zeige Tabellen-Informationen"""
    
    manager = get_table_manager(table_name)
    try:
        total = manager.count()
        
        # Hole Sample für Spalten-Info
        sample = manager.get_all(1)
        if sample:
            df = manager.to_dataframe(sample)
            columns = len(df.columns)
            column_names = df.columns.tolist()
        else:
            columns = 0
            column_names = []
        
        click.echo(f"ℹ️ {table_name.upper()} Information:")
        click.echo("=" * 40)
        click.echo(f"Gesamt Einträge: {total:,}")
        click.echo(f"Spalten: {columns}")
        
        if column_names:
            click.echo(f"\nVerfügbare Spalten:")
            for i, col in enumerate(column_names, 1):
                click.echo(f"  {i:2d}. {col}")
        
    finally:
        manager.close()

@cli.command()
@click.argument('table_name')
@click.argument('sql_query')
@click.option('--output', '-o', help='Exportiere Ergebnis zu CSV')
def query(table_name, sql_query, output):
    """⚡ Führe einfache SQL-Abfrage aus"""
    
    try:
        # Sicherheitscheck - nur SELECT erlauben
        if not sql_query.strip().upper().startswith('SELECT'):
            raise click.ClickException("Nur SELECT-Abfragen erlaubt")
        
        df = pd.read_sql(sql_query, engine)
        
        if output:
            df.to_csv(output, index=False)
            click.echo(f"✅ {len(df)} Ergebnisse exportiert nach {output}")
        else:
            click.echo(f"⚡ SQL Ergebnis ({len(df)} Zeilen):")
            click.echo("=" * 60)
            click.echo(df.to_string(index=False))
        
    except Exception as e:
        raise click.ClickException(f"SQL Fehler: {e}")

@cli.command()
@click.argument('table_name')
@click.option('--taxonomy', help='Filter: Taxonomy Code')
@click.option('--template', help='Filter: Template Code')
@click.option('--module', help='Filter: Module Code')
@click.option('--datapoint', help='Filter: Datapoint ID')
@click.option('--ko', help='Filter: Konzept Code')
@click.option('--limit', '-l', default=50, help='Max Ergebnisse')
@click.option('--export', '-e', help='Exportiere zu CSV')
def filter(table_name, taxonomy, template, module, datapoint, ko, limit, export):
    """🔧 Multi-Filter Suche (kombiniert mehrere Kriterien)"""
    
    manager = get_table_manager(table_name)
    try:
        if table_name == 'its':
            query = manager.session.query(ITSBaseData_new)
            
            # Sammle aktive Filter
            active_filters = []
            filter_info = []
            
            if taxonomy:
                query = query.filter(ITSBaseData_new.taxonomy_code.contains(taxonomy))
                active_filters.append(f"taxonomy: {taxonomy}")
            
            if template:
                query = query.filter(ITSBaseData_new.template_code.contains(template))
                active_filters.append(f"template: {template}")
            
            if module:
                query = query.filter(ITSBaseData_new.module_code.contains(module))
                active_filters.append(f"module: {module}")
            
            if datapoint:
                try:
                    query = query.filter(ITSBaseData_new.datapoint == int(datapoint))
                    active_filters.append(f"datapoint: {datapoint}")
                except ValueError:
                    click.echo(f"❌ '{datapoint}' ist keine gültige Datapoint ID")
                    return
            
            if ko:
                query = query.filter(ITSBaseData_new.ko.contains(ko))
                active_filters.append(f"konzept: {ko}")
            
            if not active_filters:
                click.echo("❌ Mindestens ein Filter erforderlich")
                click.echo("💡 Beispiel: python db_cli.py filter its --taxonomy FINREP --template 'F 01.01'")
                return
            
            entries = query.limit(limit).all()
        else:
            click.echo(f"❌ Multi-Filter für Tabelle '{table_name}' noch nicht implementiert")
            return
        
        if not entries:
            filter_str = ", ".join(active_filters)
            click.echo(f"❌ Keine Ergebnisse für Filter: {filter_str}")
            return
        
        df = manager.to_dataframe(entries)
        
        # Export oder Anzeige
        if export:
            df.to_csv(export, index=False)
            click.echo(f"✅ {len(df)} gefilterte Einträge exportiert nach: {export}")
        
        # Anzeige
        filter_str = ", ".join(active_filters)
        click.echo(f"🔧 Multi-Filter Ergebnisse ({filter_str}): {len(entries)} gefunden")
        click.echo("=" * 100)
        
        # Zeige relevante Spalten
        cols = ['id', 'datapoint', 'ko', 'taxonomy_code', 'template_code', 'module_code']
        available_cols = [col for col in cols if col in df.columns]
        display_df = df[available_cols] if available_cols else df
        
        click.echo(display_df.to_string(index=False))
        
        if len(entries) == limit:
            click.echo(f"\n⚠️  Ergebnisse auf {limit} begrenzt. Verwenden Sie --limit 0 für alle oder --export für vollständigen Export")
        
    finally:
        manager.close()

@cli.command()
@click.argument('table_name')
def fields(table_name):
    """📝 Zeige alle durchsuchbaren Felder einer Tabelle"""
    
    if table_name == 'its':
        searchable_fields = [
            ('ko', 'Konzept Code (z.B. ISFIN0000001)'),
            ('datapoint', 'Datapoint ID (numerisch)'),
            ('taxonomy_code', 'Taxonomy Code (z.B. FINREP_3.2.1)'),
            ('template_code', 'Template Code (z.B. F 01.01)'),
            ('template_label', 'Template Bezeichnung'),
            ('module_code', 'Module Code (z.B. FINREP9)'),
            ('table_code', 'Table Code'),
            ('table_name', 'Tabellen Name'),
            ('x_axis_rc_code', 'X-Achse Code'),
            ('x_axis_name', 'X-Achse Name'),
            ('y_axis_rc_code', 'Y-Achse Code'),
            ('y_axis_name', 'Y-Achse Name'),
            ('z_axis_rc_code', 'Z-Achse Code'),
            ('z_axis_name', 'Z-Achse Name'),
            ('criteria', 'Kriterien')
        ]
        
        click.echo(f"📝 Durchsuchbare Felder in {table_name.upper()}:")
        click.echo("=" * 60)
        for field, description in searchable_fields:
            click.echo(f"  {field:<18} - {description}")
        
        click.echo("\n💡 Einzelne Feld-Suche:")
        click.echo("   python db_cli.py search its 'suchterm' --field ko")
        click.echo("   python db_cli.py konzept ISFIN0000001")
        
        click.echo("\n🔧 Multi-Filter Suche:")
        click.echo("   python db_cli.py filter its --taxonomy FINREP --template 'F 01.01'")
        click.echo("   python db_cli.py filter its --ko ISFIN --module FINREP9")
        
    else:
        click.echo(f"❌ Felder für Tabelle '{table_name}' noch nicht definiert")

if __name__ == '__main__':
    cli()