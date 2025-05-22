class Node:
    def __init__(self, tablecode, componenttypename, componentcode, componentlabel,
                 headerflag, level, order, taxonomycode, displaybeforechildren,
                 ordinateid, parentordinateide):
        self.tablecode = tablecode
        self.componenttypename = componenttypename
        self.componentcode = componentcode
        self.componentlabel = componentlabel
        self.headerflag = headerflag
        self.level = level
        self.order = order
        self.taxonomycode = taxonomycode
        self.displaybeforechildren = displaybeforechildren
        self.ordinateid = ordinateid
        self.parentordinateide = parentordinateide

        self.parent = None   # Wird später gesetzt
        self.children = []   # Liste für Kind-Knoten

    def get_path_labels(self):
        """Gibt die ComponentLabels vom aktuellen Knoten bis zur Wurzel zurück."""
        path = []
        current = self
        while current is not None:
            path.append(current.componentlabel)
            current = current.parent
        return path

    def get_path_codes(self):
        """Gibt die ComponentCodes vom aktuellen Knoten bis zur Wurzel zurück."""
        path = []
        current = self
        while current is not None:
            path.append(current.componentcode)
            current = current.parent
        return path
    
    def get_level_codes(self):
        """Gibt die ComponentCodes vom aktuellen Knoten bis zur Wurzel zurück."""
        path = []
        current = self
        while current is not None:
            path.append(current.level)
            current = current.parent
        return path
    
    def get_component_label(self):
        path = []
        current = self
        while current is not None:
            path.append(current.componentlabel)
            current = current.parent
        return path
    

    def __repr__(self):
        return f"Node({self.ordinateid}, {self.componentlabel})"