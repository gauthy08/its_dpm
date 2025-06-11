from sqlalchemy import create_engine, Column, Integer, String, Date, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

class Konzept(Base):
    __tablename__ = 'Konzepte'

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String, nullable=False)
    smart_cube_konzept = Column(String)
    pflichtkonzept = Column(Boolean)
    dimensionskombination = Column(String)
    dimensionen = Column(String)
    konzepttyp = Column(String)
    observ_schluesselgruppe = Column(String)
    scs_einschraenkung = Column(String)
    anubis_rechenregel = Column(String)
    aggregationstyp = Column(String)
    kurzbezeichnung = Column(String)
    kurzbezeichnung_englisch = Column(String)
    bezeichnung = Column(String)
    bezeichnung_englisch = Column(String)
    beschreibung = Column(String)
    gueltig_von = Column(Date)
    gueltig_bis = Column(Date)
    mdi_relevant = Column(Boolean)
    mdi_modellierungstyp = Column(String)
    erhebungsteile = Column(String)

class ITSBaseData(Base):
    __tablename__ = 'ITS_base_data'

    id = Column(Integer, primary_key=True, autoincrement=True)  # Optional: Ein Primärschlüssel, falls gewünscht
    datapoint = Column(Integer)
    konzept_code = Column(String)
    taxonomy_code = Column(String)
    template_id = Column(String)
    template_label = Column(String)
    module_id = Column(String)
    module_gueltig_von = Column(DateTime)
    module_gueltig_bis = Column(DateTime)
    table_id = Column(String)
    table_name = Column(String)
    criteria = Column(String)
    x_axis_rc_code = Column(String)
    x_axis_name = Column(String)
    y_axis_rc_code = Column(String)
    y_axis_name = Column(String)
    z_axis_rc_code = Column(String)
    z_axis_name = Column(String)
    
class ITSBaseData_new(Base):
    __tablename__ = 'ITSBaseData_new'

    id = Column(Integer, primary_key=True, autoincrement=True)  # Optional: Ein Primärschlüssel, falls gewünscht
    datapoint = Column(Integer)
    ko = Column(String)
    taxonomy_code = Column(String)
    template_code = Column(String)
    template_label = Column(String)
    module_code = Column(String)
    module_gueltig_von = Column(DateTime)
    module_gueltig_bis = Column(DateTime)
    table_code = Column(String)
    table_name = Column(String)
    criteria = Column(String)
    x_axis_rc_code = Column(String)
    x_axis_name = Column(String)
    y_axis_rc_code = Column(String)
    y_axis_name = Column(String)
    z_axis_rc_code = Column(String)
    z_axis_name = Column(String)
    
    
class DPM_datapoint(Base): 
    __tablename__ = 'DPM_data_points'
    
    id = Column(Integer, primary_key=True, index=True)
    datapoint_vid = Column(Integer)
    datapoint_id = Column(Integer)
    dimension_label = Column(String)
    member_name = Column(String)
    
class Template_Finrep(Base):
    __tablename__ = 'Template_Finrep'
    id = Column(Integer, primary_key=True, index=True)
    template_sheet = Column(Integer)
    table = Column(String)
    axis = Column(String)
    coord = Column(String)
    text = Column(String)
    reference = Column(String)
    extra = Column(String)
    
class MergedData(Base): 
    __tablename__ = 'MergedData'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    konzept_id = Column(Integer, ForeignKey('Konzepte.id'))
    its_base_data_id = Column(Integer, ForeignKey('ITS_base_data.id'))
    datapoint = Column(Integer)
    konzept_code = Column(String)
    taxonomy_code = Column(String)
    template_id = Column(String)
    template_label = Column(String)
    module_id = Column(String)
    module_gueltig_von = Column(DateTime)
    module_gueltig_bis = Column(DateTime)
    table_id = Column(String)
    table_name = Column(String)
    criteria = Column(String)
    x_axis_rc_code = Column(String)
    x_axis_name = Column(String)
    y_axis_rc_code = Column(String)
    y_axis_name = Column(String)
    z_axis_rc_code = Column(String)
    z_axis_name = Column(String)
    datapoint_vid = Column(Integer)
    #datapoint_id = Column(Integer)
    dimension_label = Column(String)
    member_name = Column(String)
    
class Finrep_Y_reference(Base): 
    __tablename__ = 'Finrep_Y_reference'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    worksheet = Column(String)
    table_ = Column(String)   # Achtung auf Namenskonflikt mit SQL
    axis = Column(String)
    coord = Column(String)
    text = Column(String)
    reference = Column(String)
    extra = Column(String)
    rag_text = Column(String)
    
class DPM_tableStructure(Base):
    __tablename__ = 'DPM_tableStructure'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    tablecode = Column(String)
    componenttypename = Column(String)   
    componentcode = Column(String)
    componentlabel = Column(String)
    headerflag = Column(String)
    level = Column(Integer)
    order = Column(Integer)
    taxonomycode = Column(String)
    displaybeforechildren = Column(String)
    ordinateid = Column(Integer)
    parentordinateide = Column(Integer)
    