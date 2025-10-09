import re
import requests
import pandas as pd

def match_cases(pattern, text, pattern2 = None):
    """
    Matches text to a pattern.
    
    :param pattern: reference pattern 
    :param text: text to be matched
    :param pattern2: (optional) second pattern if the first one fails
    (for example there can be either "Point (1) of Article 34" or "Article 34(15)(1)" and it matches the (1)) 

    :return: the relevant group corresponsing to the pattern, else None
    
    """
    if pattern.match(text):
        return pattern.match(text).group(1)
    elif pattern2 and pattern2.match(text):
        return pattern2.match(text).group(1)
    else:
        return slice(None)


def get_info(ref_list):
    """
    Extracts the information from each reference. 
    "Point (i)(a) of Article 43(1) CRR" -> (Article: 43, Paragraph: (1), Point: (a), Subpoint: (i))
    
    :param ref_list: list of all references
    :return row_list: list of all information in the form of tuples (43, 1, a, i)
    
    """
    row_list = []

    # pattern regex initialize
    paragraphMatch_before = re.compile(r"Paragraph (.*) of")
    numberMatch = re.compile(r"Point (\(\d+\))")
    pointMatch = re.compile(r"Point (\([a-zA-Z]+\))")
    subpointMatch = re.compile(r"Point \(\w+\)(\([a-zA-Z]+\))")
    articleMatch = re.compile(r".*Article\s*(\d+\w?)")
    paragraphMatch_after = re.compile(r".*Article\s*\d+\w?\s*\((\d+)\)")
    numberMatch_after = re.compile(r".*Article\s*\d+\w?\s*\(\d+\)(\s*\(\d+\))")

    for ref in ref_list:
        if "CRR" not in ref:
            continue
        article = match_cases(articleMatch, ref)
        paragraph = match_cases(paragraphMatch_before,ref, pattern2=paragraphMatch_after)
        number = match_cases(numberMatch, ref, pattern2=numberMatch_after)
        point = match_cases(pointMatch, ref)
        subpoint = match_cases(subpointMatch, ref)

        if paragraph != slice(None) and paragraph[-1] != ".":
            paragraph += "."
            
        row_list.append((article, paragraph, number, point, subpoint))

    return row_list

        
def handle_indices(indx_list, df):
    """
    Handles how the indices are saved - removed dots etc.
    Also handles (i) cases - it can be either a subpoint or point.
    If the previous point was (h) then it is most likely a point. If not it should be a subpoint.
    
    :param indx_list: the get_info result, the list of info for all references
    :param df: CRR dataframe

    :return df_list: list of cleaned indices
    """
    df_list = []
    
    for x in indx_list:
        try:
            x_df = df.loc[x]
    
            if x_df.empty and x[3] == "(i)":
                x_df = df.loc[(x[0], x[1], x[2], "(h)", "(i)")]
            elif x_df.empty and x[1] != slice(None):
                x_df = df.loc[(x[0], slice(None), f"({x[1].strip('.')})", x[3], x[4])]
            df_list.append(x_df)
        except KeyError:
            df_list.append(pd.DataFrame())
    
    return df_list


def get_text(annex_df, crr_df):
    """
    Extract text excerpts based on the row information.
    
    :param annex_df: dataframe, containing the rows/columns and their legal references
    :param crr_df: dataframe, containing the texts

    :return full_text: end result text of all points mentioned in the legal references
    
    """
    full_text = []
    for ref in annex_df["Legal References"]:
        if not pd.isna(ref):
            reflist = re.sub("\n", "", ref).split(";")
            reflist_indexform = get_info(reflist)
            dfs = handle_indices(reflist_indexform, crr_df)
            for count, df in enumerate(dfs):
                if not df.empty:
                    df.iloc[0,-1] = f"Article {reflist_indexform[count][0]}\n" + df.iloc[0]["text"]
            if dfs:
                conc = pd.concat(dfs)
                conc = conc.drop_duplicates()
                if conc.empty:
                    full_text.append("")
                    continue
                text = "\n".join(conc["text"].dropna().to_list())
                full_text.append(text)
                continue
            full_text.append("")
        else:
            full_text.append("")
    return full_text


def chat_with_openwebui(token, model, query, web_ui_base_url, type_id = None, type_knowledge = None): 
    '''
    Relays the prompt to the chatbot model via openwebui API endpoint.
    
    :param type_knowledge: can be 'collection' or 'file'
    :param type_id: the id to the knowledge collection/file
    :param token: the web_ui_token
    :param query: the full prompt
    :param model: the model name
    :param web_ui_base_url: the url to the webui 

    :return: the response of the chatbot
     
    '''
    url = f'{web_ui_base_url}/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': query}]
    }

    if type_knowledge is not None:
        payload['files'] = [{'type': type_knowledge, 'id': type_id}]


    response = requests.post(url, headers=headers, json=payload, timeout=60)
    return response.json()

    
def get_id_for_knowledge(get_knowledge:dict, name:str) -> str:
    """
    Retrieves the ID of a knowledge base by its name.
    
    :param get_knowledge: The response from the server containing the list of knowledge bases.
    :param name: The name of the knowledge base to search for.
    
    :return: The ID of the knowledge base if found, otherwise None.
    
    """
    for item in get_knowledge:
        if item['name'] == name:
            return item['id']
    return None


def get_knowledge(token, web_ui_base_url) -> dict:
    """
    Retrieves the list of knowledge bases from the OpenWebUI server.
    
    :param token: The API token for authentication.
    :return: The response from the server as a dictionary.
    
    """
    url = f'{web_ui_base_url}api/v1/knowledge/list'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    return response.json()


def list_all_knowledge(get_knowledge:dict) -> None:
    """
    Prints all available knowledge ids.

    :param get_knowledge: the list of knowledge bases from get_knowledge()
    :return: None
    """
    for item  in get_knowledge:
        print(item['name'], item['id'])


def get_id_for_knowledge(get_knowledge:dict, name:str) -> str:
    """
    Retrieves the ID of a knowledge base by its name.
    
    :param get_knowledge: The response from the server containing the list of knowledge bases.
    :param name: The name of the knowledge base to search for.
    
    :return: The ID of the knowledge base if found, otherwise None.
    """
    for item in get_knowledge:
        if item['name'] == name:
            return item['id']
    return None

def prompt_setup(context, question):
    """
    Combines the context and the questions and adds them to the prompt.

    :param context: the knowledge context used as reference
    :param question: the desired question

    :return: the full prompt
    
    """
    prompt = f'''You are a helpful AI document and information assistant. 
    Use your data and the context below to answer the question. Please follow these rules:
    1. Provide a clear and compact response without redundancies. Add citations when necessary.
    2. The answer should start with: '[term]' refers to. Please use an appropriate [term].
    3. If you do not know the answer, respond with: "I can't find the final answer".
    4. Do not include any descriptive sentences such as 'This definition is consistent...' or anything about 'context' or if somethind is 'not defined'.
    5. Do not include any first person sentences such as 'I can't ...' except rule 3.

        
    Context: {context}\n
    Question: {question}\n
    Answer:
    '''
    return prompt

def chatbot_case(prompt, web_ui_token, model_name, web_ui_base_url, knowledge_id = "", pdf=False):
    """
    Wrapper for the chatbot use.

    :param prompt: the full prompt to be relayed to the chatbot
    :param pdf: True/False - whether to use the knowledge pdf or not, default is False 
                            !!! if True knowledge_id has to specified as well
    :param web_ui_token: the web ui token
    :param model_name: the model name
    :param web_ui_base_url: the url to the webui 


    :return: the chatbot response content
    """
    if pdf:
        response = chat_with_openwebui(web_ui_token, model_name, prompt, web_ui_base_url, type_id = knowledge_id, type_knowledge='collection')
        return response["choices"][0]["message"]["content"]
    else:
        response = chat_with_openwebui(web_ui_token, model_name, prompt, web_ui_base_url)
        return response["choices"][0]["message"]["content"]