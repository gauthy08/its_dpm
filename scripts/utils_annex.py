import docx
import pandas as pd
import re
import unicodedata

"""
RELEVANT REGEX RULES

    (1) r"(?:(?:[Pp]oints?|[Pp]aragraphs?)(?:\s*[\(\d].*?of ))?"   -   extracts "Point/ Paragraph of" in a text 
    (2) r"(?:\d+[a-z]*(?:\s*\([a-z0-9]+\)){,2})"                   -   extracts the number of an article like "43", "43a", "43(1)" in "Article 43a" etc
    (3) r"(?:(?:, | and | to )" + number_ref + r")*"               -   extracts any addition like "and 43" in "Article 4 and 43" 
    (4) r"(?:\s*\b[A-Z]+\b)?(?: of (?:(?:Commission )?(?:Delegated )?Regulation (?:\(EU\)\s*?)|Directive)?(?:\s*No\s*)?(?:\w+\/\w+(?:\/\w+)?))?"
                                                                   -   extracts the regulations like CRR, Directive No 23/2014
"""


def get_content_dict(doc):
    """
    Extract all tables from a word document and their corresponding titles in a dictionary.
    Example (key,value) pair: C 01.00 - OWN FUNDS (CA1) : [<docx.table.Table object at 0x7fd8443c2ec0>]

    :param doc: a word document loaded with docx
    
    :return content_dict: dictionary (title, table)
    
    """
    # Initialize empty dictionary, table_name: table address
    content_dict = {}

    # Iterate through the document
    for child in doc.iter_inner_content():
        # search only for the paragraphs with the title style and starting with C 0.4 etc
        if child.style.name == 'Instructions Überschrift 2' and re.search("C\s\d+\.\d+",child.text):
            # remove the numbering in front of a name
            table_name_temp = re.sub("^(?:\d+\.?)+\s+","",child.text)
            content_dict[table_name_temp] = []
        elif isinstance(child, docx.table.Table):
            content_dict[table_name_temp].append(child)

    return content_dict


def get_text_and_title(text):
    """
    Splits the text into Text ohne Überschrift, Überschrift, Überschrift Ziffern, Überschrift Text

    :param text: full text containing the title and the legal references
    
    :return: list with the relevant splitted text [textohne, title, title_ziffern, title_text]
    
    """
    text = unicodedata.normalize("NFKD", text)
    text_list = text.split("\n")

    # Text ohne Überschrift
    textohne = "\n".join(text_list[1:])
    # Überschrift
    title = re.sub(r"[\t\s]+", " ", text_list[0])
    # Überschrift Ziffern
    title_ziffern = re.match(r"(?:\d+\.*)+", title)
    if title_ziffern:
        title_ziffern = title_ziffern.group(0)
        # Überschrift Text
        title_text = title[len(title_ziffern)+1:]
    else:
        title_ziffern = ""
        title_text = title
 
    return [textohne, title, title_ziffern, title_text]


def add_missing_references(reflist):
    """
    Adds the missing references (CRR, BAD, etc) retrospectively. Very depedent on the read in order, assumes 
    that the last captured reference is valid for all the instances before it.
    [Article 43, Article 87 CRR, Article 98, Article 09 BAD] --> 
                                         [Article 43 CRR, Article 87 CRR, Article 98 BAD, Article 09 BAD]
    
    :param reflist: list containing the articles in read-in order
    
    :return new_reflist: articles list with the added references
    
    """
    # regex for extracting the reference Article 45  CRR -> CRR
    regulations_ref = re.compile(r"(?:\s*\b[A-Z]+\b)|(?: of (?:(?:Commission )?(?:Delegated )?Regulation (?:\(EU\)\s*?)|Directive)?(?:\s*No\s*)?(?:\w+\/\w+(?:\/\w+)?))")
    
    new_reflist = []
    current_valid_ref = ""
    
    for ref in reversed(reflist):
        regulation = re.search(regulations_ref, ref)
        
        if regulation:
            current_valid_ref = regulation.group()
        else:
            ref += current_valid_ref
            
        if ref not in new_reflist:
            new_reflist.insert(0,ref)

    for indx, ref in enumerate(new_reflist):
        regulation_check = re.search(regulations_ref, ref)

        if not regulation_check:
            new_reflist[indx] += current_valid_ref      
            
    return new_reflist


def clean_up_references(ref_list):
    """
    Unifies and cleans up the extracted articles. 
    "Point 3,4,5 of Articles 45 to 64 and 12 CRR" -> "Point 3 of Article 45 CRR", "Point 3 of Article 46 CRR", etc

    :param ref_list: extracted articles list
    
    :returns cleaned_up_list: cleaned articles list
    
    """
    matchCases = re.compile(r"((?:Points*)|(?:Paragraphs*))?\s*(?:(.*) of )?Articles?\s*(\d+\w?)(\(\d+\w?\))?(\(\d+\))?\s*(.*)")
    regulations_ref = re.compile(r"(?:\s*\b[A-Z]+\b)|(?: of (?:(?:Commission )?(?:Delegated )?Regulation (?:\(EU\)\s*?)|Directive)?(?:\s*No\s*)?(?:\w+\/\w+(?:\/\w+)?))")
    
    cleaned_up_list = []

    for ref in ref_list:
        new_cleaned_up_list = []
        
        ref = re.sub(r'[Aa]rticles?', r'Article', ref)
        ref = re.sub(r'[Pp]oints?', r'Point', ref)
        ref = re.sub(r'[Pp]aragraphs?', r'Paragraph', ref)
        ref = re.sub(r",\s*", " and ", ref)
        ref = re.sub(r"\s*or\s*", " and ", ref)

        regulation = re.search(regulations_ref, ref)
        regulation_text = ""
        if regulation:
            ref = re.sub(regulations_ref, "", ref)
            regulation_text = regulation.group()

        casesMatch = matchCases.match(ref)
        currentArticle = casesMatch.group(3) + (casesMatch.group(4) or "") + (casesMatch.group(5) or "") 
        
        # search for stuff BEFORE article
        if casesMatch and casesMatch.group(2):
            for point in casesMatch.group(2).split(" and "):
                if re.match(r"\(\d+\)", point) and casesMatch.group(1).title() == "Paragraph":
                    point = re.sub(r"[\(\)]","",point)
                if "to" in point:
                    to_list = re.split(r'\s*to\s*', point)
                    to_list = [re.sub(r"[\(\)]","",to) for to in to_list]
                    if re.match(r"\d+\w", to_list[0]):
                        new_articles = [to_list[0][:-1]+chr(x) for x in range(ord(to_list[0][-1])+1, ord(to_list[1][-1])+1)]
                    elif re.match(r"[a-zA-Z]", to_list[0]):
                        new_articles = ["("+chr(x)+")" for x in range(ord(to_list[0]), ord(to_list[1])+1)]
                    elif casesMatch.group(1).title() == "Paragraph":        
                        new_articles = [str(x) for x in range(int(to_list[0])+1, int(to_list[1])+1)]
                    else:
                        new_articles = ["("+str(x)+")" for x in range(int(to_list[0])+1, int(to_list[1])+1)]

                    for newart in new_articles:
                        refStr = f"{casesMatch.group(1).title()} {newart} of Article {currentArticle}{regulation_text}"
                        if refStr not in cleaned_up_list:
                            cleaned_up_list.append(refStr)
                    continue  

                    
                refStr = f"{casesMatch.group(1).title()} {point} of Article {currentArticle}{regulation_text}"
                if refStr not in cleaned_up_list:
                    cleaned_up_list.append(refStr)
            continue

        # search for stuff AFTER article
        if casesMatch and casesMatch.group(6):
            and_list = re.split(r'\s*and\s*', re.sub(r'^and\s*', '', casesMatch.group(6)))
            for point in and_list:
                if "to" in point:
                    to_list = re.split(r'\s*to\s*', re.sub(r'^to\s*', '', point.strip()))
                    if len(to_list) == 1:
                        if re.match("\d+\w", casesMatch.group(3)):
                            new_articles = [casesMatch.group(3)[:-1]+chr(x) for x in range(ord(casesMatch.group(3)[-1])+1, ord(to_list[0][-1])+1)]
                        else:
                            new_articles = [x for x in range(int(casesMatch.group(3))+1, int(to_list[0])+1)]
                    else:
                        if re.match("\d+\w", to_list[0]):
                            new_articles = [to_list[0][:-1]+chr(x) for x in range(ord(to_list[0][-1])+1, ord(to_list[1][-1])+1)]
                        else:        
                            new_articles = [x for x in range(int(to_list[0])+1, int(to_list[1])+1)]

                    for newart in new_articles:
                        refStr = f"Article {newart}{regulation_text}"
                        if refStr not in cleaned_up_list:
                            cleaned_up_list.append(refStr)
                    continue
                            
                refStr = f"Article {point}{regulation_text}"
                if refStr not in cleaned_up_list:
                    cleaned_up_list.append(refStr)

        refStr = f"Article {currentArticle}{regulation_text}"
        if refStr not in cleaned_up_list:
            cleaned_up_list.append(refStr)
                
    return cleaned_up_list
        
        
def get_legal_references(text):
    """
    Extracts the legal references from the full text. Matches everything between the words Point/ Paragraph and Article
    as well as everything between Article and the reference CRR/BED etc
    
    Example Text: Point (118) of Article 4(1) and Article 72 CRR
                  The own funds of an institution shall consist of the sum of its Tier 1 capital and Tier 2 capital.
                  
    Extracted Refernces: Point (118) of Article 4(1) CRR
                         Article 72 CRR
                         
    :param text: the text containing all information
    
    :returns ref_list: the list of the extracted references
    
    """
    front_ref = r"(?:(?:[Pp]oints?|[Pp]aragraphs?)(?:\s*[\(\d].*?of ))?"
    number_ref = r"(?:\d+[a-z]*(?:\s*\([a-z0-9]+\)){,2})"
    additions_ref = r"(?:(?:, | and | to )" + number_ref + r")*"
    regulations_ref = r"(?:\s*\b[A-Z]+\b)?(?: of (?:(?:Commission )?(?:Delegated )?Regulation (?:\(EU\)\s*?)|Directive)?(?:\s*No\s*)?(?:\w+\/\w+(?:\/\w+)?))?"

    full_pattern = front_ref + r"Articles?\s*" + number_ref + additions_ref + regulations_ref
    
    ref_list = re.findall(full_pattern, text)
    ref_list = clean_up_references(ref_list)
    ref_list = add_missing_references(ref_list)
    
    return ref_list


def get_row_info(trow, name, row_column):
    """
    Summarizes any row from the table in a dictionary with columns:
    Table | Text | Text ohne Überschrift | Legal References | Überschrift | Überschrift Ziffern | Überschrift Text
                         
    :param trow: table row address
    :param name: extracted table name
    :param row_column: string indicating if it is a COLUMN or ROW
    
    :returns ref_list: the list of the extracted references
    
    """
    
    dict_template = {"Table":None, row_column:None, "Text":None, "Text ohne Überschrift":None, 
                     "Legal References":None, "Überschrift":None, "Überschrift Ziffern":None, 
                     "Überschrift Text":None}
    
    dict_template["Table"] = name.strip()
    dict_template[row_column] = trow.cells[0].text
    dict_template["Text"] = trow.cells[1].text

    text_list = get_text_and_title(trow.cells[1].text)
    dict_template["Text ohne Überschrift"] = text_list[0]
    dict_template["Überschrift"] = text_list[1]
    dict_template["Überschrift Ziffern"] = text_list[2]
    dict_template["Überschrift Text"] = text_list[3]

    legalref_text = get_legal_references(text_list[0])
    legalref_title = get_legal_references(text_list[1])

    legalref_list = legalref_text + [x for x in legalref_title if x not in legalref_text]
    dict_template["Legal References"] = ";\n".join(legalref_list)

    return dict_template


def extract_metadata(con_dict):
    """
    Extracts the tables from the word document and summarizes them into Data Frames of the form:
    Table | Text | Text ohne Überschrift | Legal References | Überschrift | Überschrift Ziffern | Überschrift Text

    :param con_dict: content dictionary created with get_content_dict of the form (table name, table)
    
    :return rowd_full, cold_full - data frames for all rows/ columns from the word document
    
    """
    # data frame dict for ROWS
    col_list_rowdf = []
    r_rows = re.compile("[Rr]ows*")
    
    # data frame dict for COLUMNS
    col_list_coldf = []
    r_cols = re.compile("[Cc]olumns*")
    
    # loop for filling the dicts with the relevant information
    for name, tables in con_dict.items():
        for table in tables:
            header = [x.text for x in table.rows[0].cells]
    
            if bool(list(filter(r_rows.match, header))):
                for row in table.rows[1:]:
                    col_list_rowdf.append(get_row_info(row, name, "Row"))
            elif bool(list(filter(r_cols.match, header))):
                for row in table.rows[1:]:
                    col_list_coldf.append(get_row_info(row, name, "Column"))

    rowdf_full = pd.DataFrame(col_list_rowdf)
    coldf_full = pd.DataFrame(col_list_coldf)

    return rowdf_full, coldf_full