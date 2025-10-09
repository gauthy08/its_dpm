import pymupdf
from collections import Counter
import re


def get_indents(file_path, ref_page = 1):
    """
    Findet die relevanten Einrückungen (links, oben), die für die Bestimmung der Absätze und Paragraphen wichtig sind.

    :param file_path: Pfad zum Dokument
    :param ref_page: Referenzseite - Seite mit Headertext und normalen Absätzen
    
    :return HEADER_TEXT_INDENT: Header-Text Einrückung
    
    """
    indents = []
    headers = []
    
    with pymupdf.open(file_path) as pdf:
        blocks = pdf[ref_page].get_text("blocks")
        for block in blocks:
            indents.append(round(block[0]))
            headers.append(round(block[1]))

    c = Counter(indents)

    # gets the header indent - the text thats positioned the highest
    HEADER_TEXT_INDENT = min(headers)

    return HEADER_TEXT_INDENT
    
        
class Extract:
    """
    Creates a class serving as container for the full document. Keeps track of everything found in a dict
    {article: 4, point: b, text: Placeholder} and updates it with new points accordingly.

    :attribute file_path: path to file
    :attribute start_page: starting page for the extraction, default: 1
    :attribute end_page: last page for the extraction, default: -1 (whole document)
    :attribute header_text_indent: position in the PDF for the header_text

    :method get_patterns(): initialize patterns
    :method get_id_dict(): reverse dictionary
    :method handle_match(text): match text to structure if the text is a point(paragraph/point/etc)
    :method handle_no_match(text): handle unassigned texts (append to previous point etc)
    :method crr_extraction(): main extraction function
    
    """
    def __init__(self, file_path, start_page = 1, end_page = -1, header_text_indent = 48):
        self.file_path = file_path
        self.start_page = start_page
        self.end_page = end_page
        self.header_text_indent = header_text_indent
        
        self.amends_pattern = re.compile(r"▼.*\n")
        self.dict_hierarchy = {0: "article", 1: "paragraph", 2: "number", 3: "point", 4: "subpoint"}
        self.results = []
        self.last_streak = None
        self.currentStructures = {"article": "", "paragraph": "", "number": "", "point": "", "subpoint": ""}

        self.patterns = self.get_patterns()
        self.dict_hierarchy_id = self.get_id_dict()
        

    def get_patterns(self):
        """
        Initialize the pattern regex.

        :return patterns: dictionary with the relevant regex for each structure.
        
        """
        patterns = {"article": re.compile(r'Article (\d+\w*)'),
                    "subpoint": re.compile(r'(\([ivx]+\))\s+([\s\S]*)$', re.IGNORECASE),
                    "point": re.compile(r'(\([a-zA-Z]+\))\s+([\s\S]*)$'),
                    "number": re.compile(r'(\(\d+\w*\))\s+([\s\S]*)$'),
                    "paragraph": re.compile(r'(\d+\w*\.(?:\d+\.)*)\s*([\s\S]*)$')}
        return patterns

    def get_id_dict(self):
        """
        Reverse the hierarchy dictionary -> turn into ("article": 0) etc

        :return dictionary (structre: ID)
        
        """
        return {value: key for key, value in self.dict_hierarchy.items()}

    def handle_match(self, text): 
        """
        Append a matched structure into a dictionary, based on the hierarchy. For each pattern
        check if there is a match and append to the current stance.

        :param text: text to be matched
        :return temp_dict: dictionary with the current structures
        
        """
        for structure, pattern in self.patterns.items():
            is_match = pattern.match(text)
            if is_match:
                self.currentStructures[structure] = is_match.group(1)
                text_content = is_match.group(2).strip() if structure != "article" else ""
                self.last_streak = self.dict_hierarchy_id[structure]
        
                for i in range(self.last_streak+1, 5):
                    self.currentStructures[self.dict_hierarchy[i]] = ""
                        
                    
                temp_dict = {key: value for key, value in self.currentStructures.items()}
                temp_dict["text"] = self.currentStructures[structure] + " " + text_content if structure != "article" else ""
                               
                return temp_dict

    def handle_no_match(self, text):
        """
        Append unassigned/ unmatched text to the previous point.

        :param text: text to be matched
        :return temp_dict: dictionary with the current structures
        
        """
        temp_dict = {"point": "", "subpoint": ""}

        if self.currentStructures["article"] != "":
            for i in range(0,3):
                temp_dict[self.dict_hierarchy[i]] = self.currentStructures[self.dict_hierarchy[i]] if self.currentStructures[self.dict_hierarchy[i]] else ""
            temp_dict["text"] = text
            return temp_dict


    def crr_extraction(self):
        """
        Function that sorts the whole document into a list of rows suitable for a DataFrame.
        Goes through a document and sorts the text into structures based on a hierarchy.
         
        :return result: list of rows containing the information
        
        """
        with pymupdf.open(self.file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                # ignore pages until the start page
                if page_number < self.start_page:
                    continue
                # optionally set an end page - extracts everything between the beginning and the end page (incl)
                if page_number - 1 == self.end_page:
                    break
     
                # text as predefined blocks
                page_text = page.get_text("blocks") or ""
                
                for block in page_text:
                    # ignores the block if its a header
                    if abs(block[1] - self.header_text_indent) <= 5:
                        continue

                    # removes the ▼M17-like pattern that is in front of some paragraphs
                    amends_match = self.amends_pattern.match(block[4])
                    cleaned_text = re.sub(self.amends_pattern, "", block[4]).strip()
                    cleaned_text = re.sub("\n", " ", cleaned_text)
                
                    # skip text blocks like "---------", ensure there is at least one word
                    if (not re.search(r'[a-zA-Z0-9]', cleaned_text)) or re.match(r'\( 1 \)[\s\S]+', cleaned_text):  
                        continue
     
                    if re.match(r'TITLE[\s\S]+', cleaned_text) or re.match(r'CHAPTER[\s\S]+', cleaned_text):
                        self.currentStructures["article"] = ""
                        continue
    
                    block_results = self.handle_match(cleaned_text)
                    
                    # ignores introduction text if an article hasnt been found
                    if self.currentStructures["article"] == "":
                        continue

                    if block_results:
                        self.results.append(block_results)
                        continue
                   
                    block_results = self.handle_no_match(cleaned_text)   
                    if block_results:
                        self.results.append(block_results)
                        continue
        return self.results