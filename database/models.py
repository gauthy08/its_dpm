from sqlalchemy import create_engine, Column, Integer, String, Date, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()

    
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
    