import requests
import os
import time

# SSL Setup für CML
os.environ["SSL_CERT_FILE"] = '/etc/ssl/certs/ca-certificates.crt'

# Konfiguration
token = 'sk-15b54c10119c45f7a45e790a109d7c8b'
base_url = 'https://chatbot-open-webui.apps.prod.w.oenb.co.at'
model_name = 'chatbot-mistral'
knowledge_id = 'aace4dfd-3f4f-46da-9936-b38dc133e3e9'  # CRR Knowledge Base

def get_knowledge_bases():
    """Alle verfügbaren Knowledge Bases abrufen"""
    url = f'{base_url}/api/v1/knowledge/list'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    return response.json()

def chat_simple(question):
    """Einfache Chat-Funktion ohne Knowledge Base"""
    url = f'{base_url}/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': question}]#,
        #'temperature' = 0
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response_time = time.time() - start_time
    
    if response.status_code != 200:
        return f"Error {response.status_code}: {response.text}", response_time
    
    result = response.json()
    return result["choices"][0]["message"]["content"], response_time

def chat_with_kb(question):
    """Chat-Funktion mit CRR Knowledge Base"""
    url = f'{base_url}/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': question}],
        'files': [{'type': 'collection', 'id': knowledge_id}]#,
        #'temperature': 0
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response_time = time.time() - start_time
    
    if response.status_code != 200:
        return f"Error {response.status_code}: {response.text}", response_time
    
    result = response.json()
    return result["choices"][0]["message"]["content"], response_time

if __name__ == "__main__":
    # Liste alle Knowledge Bases auf
    print("📚 Verfügbare Knowledge Bases:")
    try:
        kb_list = get_knowledge_bases()
        for kb in kb_list:
            status = "✅ Aktiv" if kb['id'] == knowledge_id else "⚪ Verfügbar"
            print(f"{status} {kb['name']} (ID: {kb['id']})")
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Knowledge Bases: {e}")
    
    print("\n" + "="*50)
    
    question = "Aus welchen Dokumenten kannst du Kontext liefern?"
    
    print("🔍 Ohne Knowledge Base:")
    answer_simple, time_simple = chat_simple(question)
    print(f"⏱️ Zeit: {time_simple:.1f} Sekunden")
    print(answer_simple)
    
    print("\n📚 Mit CRR Knowledge Base:")
    answer_kb, time_kb = chat_with_kb(question)
    print(f"⏱️ Zeit: {time_kb:.1f} Sekunden")
    print(answer_kb)
    
    print(f"\n📊 Vergleich:")
    print(f"Ohne KB: {time_simple:.1f}s")
    print(f"Mit KB:   {time_kb:.1f}s")
    print(f"Overhead: {time_kb - time_simple:.1f}s ({((time_kb/time_simple - 1) * 100):.0f}% langsamer)")