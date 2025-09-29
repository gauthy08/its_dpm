import pickle
import pandas as pd
import os
import requests
import json
from fastprogress.fastprogress import progress_bar
from datetime import datetime
from pathlib import Path
from prompts import get_prompt_template, get_prompt_info, list_available_prompts


# Enhanced Production Configuration with optional LLM parameters
CONFIG = {
    'web_ui_token': 'sk-15b54c10119c45f7a45e790a109d7c8b',
    'model_name': 'chatbot-mistral',
    'web_ui_base_url': 'https://chatbot-open-webui.apps.prod.w.oenb.co.at/',
    'knowledge_id': 'aace4dfd-3f4f-46da-9936-b38dc133e3e9',  # ITS AI USE CASE (COREP)


    # 🆕 ULTRA CONSERVATIVE LLM parameters - Anti-Hallucination
    'llm_parameters': {
    "temperature": 0.0,
    "max_tokens": 100,
    "top_p": 0.5,
    "presence_penalty": 0.3,
    "frequency_penalty": 0.3
  },
    
    
    # System parameters
    'batch_size': 20,
    'output_dir': 'evaluation_experiments',
    'timeout': 60
}

PRODUCTION_PROMPT_old = """You are a precise regulatory reporting assistant specialized in interpreting ITS templates and CRR-related data points. Your task is to explain data points using ONLY the provided context from your knowledge base.

STRICT RULES:
1. Start your answer with: '{node_label}' refers to ...
2. Use ONLY the content from the provided context below – do not rely on general knowledge or make assumptions.
3. If the context does not contain sufficient information to answer, respond with: "I cannot find sufficient information in the knowledge base."
4. Do NOT add interpretations, examples, assumptions, or general banking knowledge.
5. Do NOT mention any regulation, article number, or source citation in the response.
6. Keep your response under 100 words.
7. Be factual, formal, and concise.

Template: {table_code}
Data Point: {node_code} – {node_label}
Context:
{hierarchical_context}

IMPORTANT: If any part of the definition is not clearly supported by the context, omit it. Do not speculate.

What does this data point mean based on the knowledge base above?"""



# Filter definitions for different production scopes
PRODUCTION_FILTERS = {
    'first_20': {
        'description': 'First 20 nodes only (testing)',
        'limit': 20,
        'tables': None
    },
    'c01_table_row': {
        'description': 'C 01.00 - Capital Adequacy - Own funds definition - Table row',
        'limit': None,
        'tables': {'C 01.00 - Capital Adequacy - Own funds definition - Table row'}
    },
    'c01_table_column': {
        'description': 'C 01.00 - Capital Adequacy - Own funds definition - Table column', 
        'limit': None,
        'tables': {'C 01.00 - Capital Adequacy - Own funds definition - Table column'}
    },
    'c01_both': {
        'description': 'Both C 01.00 Table row and column',
        'limit': None,
        'tables': {
            'C 01.00 - Capital Adequacy - Own funds definition - Table row',
            'C 01.00 - Capital Adequacy - Own funds definition - Table column'
        }
    },
    'all_corep': {
        'description': 'All COREP tables and components',
        'limit': None,
        'tables': None
    }
}

class ProductionFileManager:
    """Manages production files with descriptive names"""
    
    def __init__(self, base_dir="evaluation_experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create production subdirectories
        self.subdirs = {
            'production': self.base_dir / 'production_runs',
            'archives': self.base_dir / 'archived_experiments'
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)
    
    def create_production_filename(self, scope, sample_size, taxonomy="COREP_3_2"):
        """Create descriptive filename for production run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        filename = f"production_{scope}_{taxonomy}_{sample_size}nodes_{timestamp}.xlsx"
        return self.subdirs['production'] / filename
    
    def save_production_run(self, results_df, run_config):
        """Save production results with enhanced metadata including LLM parameters"""
        
        filename = self.create_production_filename(
            run_config['scope'],
            len(results_df),
            run_config.get('taxonomy', 'COREP_3_2')
        )
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main results
            results_df.to_excel(writer, sheet_name='Results', index=False)
            
            # Enhanced production metadata with LLM parameters
            llm_params = run_config.get('llm_parameters') or {}
            metadata_df = pd.DataFrame([
                ['Production Run', run_config.get('run_id', 'auto')],
                ['Created', datetime.now().isoformat()],
                ['Scope', run_config['scope']],
                ['Scope Description', run_config.get('scope_description', '')],
                ['Prompt Used', 'CONCISE (Production Standard)'],
                ['Model', run_config.get('model_name', CONFIG['model_name'])],
                ['Knowledge Base', 'ITS AI USE CASE (COREP)'],
                ['Total Nodes', len(results_df)],
                ['Taxonomy', run_config.get('taxonomy', 'COREP_3_2')],
                ['Processing Time', run_config.get('processing_time', 'Unknown')],
                ['Batch Size', run_config.get('batch_size', CONFIG['batch_size'])],
                ['Success Rate', f"{len(results_df[~results_df['chatbot_response'].str.startswith('API Error')])}/{len(results_df)}"],
                ['Notes', run_config.get('notes', '')],
                ['', ''],
                ['=== LLM PARAMETERS ===', ''],
                ['Temperature', llm_params.get('temperature', 'Standard')],
                ['Max Tokens', llm_params.get('max_tokens', 'Standard')],
                ['Top P', llm_params.get('top_p', 'Standard')],
                ['Presence Penalty', llm_params.get('presence_penalty', 'Standard')],
                ['Frequency Penalty', llm_params.get('frequency_penalty', 'Standard')]
            ], columns=['Parameter', 'Value'])
            metadata_df.to_excel(writer, sheet_name='Run_Info', index=False)
            
            # Summary statistics
            if 'response_length' in results_df.columns:
                stats_data = []
                stats_data.append(['Total Responses', len(results_df)])
                stats_data.append(['Successful Responses', len(results_df[~results_df['chatbot_response'].str.startswith(('API Error', 'Error'))])])
                stats_data.append(['Avg Response Length', results_df['response_length'].mean()])
                stats_data.append(['Min Response Length', results_df['response_length'].min()])
                stats_data.append(['Max Response Length', results_df['response_length'].max()])
                stats_data.append(['Avg Processing Time', results_df['processing_time_seconds'].mean()])
                
                stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        print(f"✅ Production run saved: {filename.name}")
        self._update_production_registry(filename, run_config)
        return filename
    
    def _update_production_registry(self, filename, run_config):
        """Update production run registry with LLM parameters"""
        registry_file = self.base_dir / "production_registry.xlsx"
        
        llm_params = run_config.get('llm_parameters') or {}
        new_entry = {
            'timestamp': datetime.now().isoformat(),
            'run_id': run_config.get('run_id', 'auto'),
            'scope': run_config['scope'],
            'scope_description': run_config.get('scope_description', ''),
            'file_name': filename.name,
            'file_path': str(filename),
            'node_count': run_config.get('node_count', 'unknown'),
            'taxonomy': run_config.get('taxonomy', 'COREP_3_2'),
            'status': 'completed',
            'temperature': llm_params.get('temperature', 'Standard'),
            'max_tokens': llm_params.get('max_tokens', 'Standard'),
            'top_p': llm_params.get('top_p', 'Standard'),
            'notes': run_config.get('notes', '')
        }
        
        if registry_file.exists():
            registry_df = pd.read_excel(registry_file)
            registry_df = pd.concat([registry_df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            registry_df = pd.DataFrame([new_entry])
        
        registry_df.to_excel(registry_file, index=False)
        print(f"📝 Production registry updated: {len(registry_df)} total runs")

def load_tree_structure(pickle_path):
    """Load tree structure from pickle file"""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def get_node_path(node):
    """Create path from root to current node"""
    path = []
    current = node
    while current is not None:
        path.insert(0, {
            'code': current.componentcode,
            'label': current.componentlabel,
            'level': current.level
        })
        current = current.parent
    return path

def create_context_from_path(path):
    """Create hierarchical context string from node path"""
    if len(path) <= 1:
        return ""
    
    context_parts = []
    for i, node in enumerate(path[:-1]):
        indent = "  " * i
        context_parts.append(f"{indent}- {node['code']}: {node['label']}")
    
    return "Hierarchical Context:\n" + "\n".join(context_parts)

# In chatbot.py - Modifikation der extract_nodes_with_filter Funktion:

def extract_nodes_with_filter(all_trees, filter_config, verbose=True):
    """Extract nodes based on production filter configuration
    
    Args:
        all_trees: Dictionary with tree structures
        filter_config: Filter configuration for nodes
        verbose (bool): If True, print processing information. Default True for backwards compatibility.
    """
    all_nodes_data = []
    
    for (table_code, comp_type), roots in all_trees.items():
        tree_key = f"{table_code} - {comp_type}"
        
        # Apply table filter if specified
        if filter_config['tables'] and tree_key not in filter_config['tables']:
            continue
            
        # Nur ausgeben wenn verbose=True
        if verbose:
            print(f"Processing: {tree_key}")
        
        def traverse_tree(node):
            path = get_node_path(node)
            context = create_context_from_path(path)
            current_node = path[-1] if path else {'code': '', 'label': '', 'level': ''}
            
            all_nodes_data.append({
                'table_code': table_code,
                'component_type': comp_type,
                'node_code': current_node['code'],
                'node_label': current_node['label'],
                'node_level': current_node['level'],
                'hierarchical_context': context,
                'full_path': ' → '.join([f"{n['code']}: {n['label']}" for n in path])
            })
            
            for child in node.children:
                traverse_tree(child)
        
        for root in roots:
            traverse_tree(root)
    
    # Apply limit if specified
    if filter_config['limit']:
        all_nodes_data = all_nodes_data[:filter_config['limit']]
    
    return pd.DataFrame(all_nodes_data)

def call_chatbot_api(prompt, config):
    """Enhanced chatbot API call with optional LLM parameter control"""
    url = f"{config['web_ui_base_url']}api/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {config['web_ui_token']}",
        "Content-Type": "application/json"
    }
    
    # Base payload
    payload = {
        "model": config['model_name'],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    # Add enhanced LLM parameters only if defined
    if config.get('llm_parameters'):
        payload.update(config['llm_parameters'])
        print(f"🔧 Using LLM parameters: {config['llm_parameters']}")
    else:
        print(f"🔧 Using Standard (default) parameters")
    
    # Add knowledge base
    if config.get('knowledge_id'):
        payload["files"] = [{'type': 'collection', 'id': config['knowledge_id']}]
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=config['timeout'])
        
        if response.status_code != 200:
            return f"API Error: HTTP {response.status_code} - {response.text[:200]}"
        
        json_response = response.json()
        
        if "choices" in json_response and json_response["choices"]:
            return json_response["choices"][0]["message"]["content"]
        else:
            return f"API Error: Unexpected response format - {list(json_response.keys())}"
        
    except requests.exceptions.RequestException as e:
        return f"API Request failed: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def test_api_connection(config):
    """Test API connection before production run"""
    print("🔍 Testing API connection...")
    test_prompt = "Test connection for production run"
    response = call_chatbot_api(test_prompt, config)
    
    if response.startswith("API Error") or response.startswith("Error"):
        print(f"❌ API Test failed: {response}")
        return False
    else:
        print(f"✅ API Test successful")
        return True

def create_production_prompt(row):
    """Create production prompt using CONCISE template"""
    return PRODUCTION_PROMPT.format(
        node_label=row['node_label'],
        node_code=row['node_code'],
        table_code=row['table_code'],
        hierarchical_context=row['hierarchical_context']
    )

def run_production_pipeline(nodes_df, config, scope):
    """Run production pipeline with enhanced parameters"""
    
    # Test API connection first
    if not test_api_connection(config):
        print("❌ Cannot proceed without working API connection")
        return None
    
    file_manager = ProductionFileManager(config['output_dir'])
    
    print(f"\n🏭 Starting production run: {scope}")
    print(f"📊 Processing {len(nodes_df)} nodes")
    if config.get('llm_parameters'):
        print(f"🔧 LLM Parameters: {config['llm_parameters']}")
    else:
        print(f"🔧 LLM Parameters: Standard (default)")
    
    results = []
    total_nodes = len(nodes_df)
    start_time = datetime.now()
    
    # Process in batches with progress tracking
    for batch_start in range(0, total_nodes, config['batch_size']):
        batch_end = min(batch_start + config['batch_size'], total_nodes)
        batch_nodes = nodes_df.iloc[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start+1}-{batch_end} ({len(batch_nodes)} nodes)")
        
        for idx, row in progress_bar(list(batch_nodes.iterrows())):
            # Create production prompt
            prompt = create_production_prompt(row)
            
            # Generate response with optional parameters
            response_start = datetime.now()
            response = call_chatbot_api(prompt, config)
            processing_time = (datetime.now() - response_start).total_seconds()
            
            # Get LLM parameters for storage
            llm_params = config.get('llm_parameters') or {}
            
            results.append({
                'node_id': f"{row['table_code']}|{row['component_type']}|{row['node_code']}",
                'node_code': row['node_code'],
                'node_label': row['node_label'],
                'table_code': row['table_code'],
                'component_type': row['component_type'],
                'hierarchical_context': row['hierarchical_context'],
                'full_path': row['full_path'],
                'production_prompt': prompt,
                'chatbot_response': response,
                'response_length': len(response),
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'batch_number': (batch_start // config['batch_size']) + 1,
                # Add LLM parameters to each result
                'llm_temperature': llm_params.get('temperature', 'Standard'),
                'llm_max_tokens': llm_params.get('max_tokens', 'Standard'),
                'llm_top_p': llm_params.get('top_p', 'Standard'),
                'llm_presence_penalty': llm_params.get('presence_penalty', 'Standard'),
                'llm_frequency_penalty': llm_params.get('frequency_penalty', 'Standard')
            })
        
        # Intermediate save after each batch
        if results:
            temp_df = pd.DataFrame(results)
            print(f"✅ Batch {batch_start // config['batch_size'] + 1} completed")
    
    # Final save
    total_time = (datetime.now() - start_time).total_seconds()
    results_df = pd.DataFrame(results)
    
    run_config = {
        'run_id': f"production_{scope}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        'scope': scope,
        'scope_description': PRODUCTION_FILTERS[scope]['description'],
        'node_count': len(results_df),
        'taxonomy': 'COREP_3_2',
        'model_name': config['model_name'],
        'batch_size': config['batch_size'],
        'processing_time': f"{total_time:.1f}s total ({total_time/len(results_df):.1f}s per node)",
        'llm_parameters': config.get('llm_parameters'),  # Can be None for Standard
        'notes': f"Production run with {'custom' if config.get('llm_parameters') else 'standard'} parameters. Scope: {scope}"
    }
    
    filename = file_manager.save_production_run(results_df, run_config)
    
    print(f"\n🎉 PRODUCTION RUN COMPLETED!")
    print(f"📊 Processed: {len(results_df)} nodes")
    print(f"⏱️ Total time: {total_time:.1f}s ({total_time/len(results_df):.1f}s per node)")
    if config.get('llm_parameters'):
        print(f"🔧 Used parameters: {config['llm_parameters']}")
    else:
        print(f"🔧 Used parameters: Standard (default)")
    print(f"📁 Results: {filename}")
    
    return results_df

def process_pickle_tree_with_chatbot(pickle_path=None, taxonomy_code="COREP_3_2"):
    """Main production function with enhanced parameter control"""
    
    # Load tree structure
    if pickle_path is None:
        pickle_path = f"tree_structures/baumstruktur_{taxonomy_code}.pkl"
    
    if not os.path.exists(pickle_path):
        print(f"❌ Pickle file not found: {pickle_path}")
        return None
    
    print(f"Loading tree structure: {pickle_path}")
    all_trees = load_tree_structure(pickle_path)
    print(f"✅ Loaded {len(all_trees)} tree combinations")
    
    # Show production options
    print(f"\n{'='*60}")
    print("PRODUCTION PIPELINE OPTIONS")
    print(f"{'='*60}")
    
    for key, filter_info in PRODUCTION_FILTERS.items():
        print(f"{key}: {filter_info['description']}")
    
    choice = input(f"\nChoose production scope: ").strip()
    
    if choice not in PRODUCTION_FILTERS:
        print(f"❌ Invalid choice. Available options: {list(PRODUCTION_FILTERS.keys())}")
        return None
    
    # Extract nodes based on selected filter
    filter_config = PRODUCTION_FILTERS[choice]
    print(f"\n🔄 Extracting nodes for: {filter_config['description']}")
    
    nodes_df = extract_nodes_with_filter(all_trees, filter_config)
    print(f"✅ Extracted {len(nodes_df)} nodes")
    
    if len(nodes_df) == 0:
        print("❌ No nodes found with selected filter!")
        return None
    
    # Show preview
    print(f"\n📋 Preview (first 3 nodes):")
    for idx, row in nodes_df.head(3).iterrows():
        print(f"  • {row['node_code']}: {row['node_label']}")
    
    # Show LLM parameters
    print(f"\n🔧 LLM Parameters to be used:")
    if CONFIG.get('llm_parameters'):
        for param, value in CONFIG['llm_parameters'].items():
            print(f"  • {param}: {value}")
    else:
        print(f"  • Standard (default parameters)")
    
    # Confirm production run
    confirm = input(f"\n🚀 Start production run for {len(nodes_df)} nodes? (y/n): ").lower()
    
    if confirm in ['y', 'yes', 'j', 'ja']:
        return run_production_pipeline(nodes_df, CONFIG, choice)
    else:
        print("❌ Production run cancelled")
        return nodes_df

# REPLACE the existing view_production_results() function in chatbot.py:
def view_production_results():
    """Enhanced view of production run results including LLM Judge statistics"""
    registry_file = Path(CONFIG['output_dir']) / "production_registry.xlsx"
    
    if not registry_file.exists():
        print("❌ No production runs found")
        return
    
    registry_df = pd.read_excel(registry_file)
    print("📋 PRODUCTION RUN HISTORY WITH JUDGE RESULTS:")
    print("=" * 80)
    
    for idx, row in registry_df.iterrows():
        print(f"\n🏭 Run {idx+1}: {row.get('scope', 'Unknown')}")
        print(f"   📅 Date: {row.get('timestamp', 'Unknown')[:16]}")
        print(f"   📊 Nodes: {row.get('node_count', 'Unknown')}")
        print(f"   🔧 Temperature: {row.get('temperature', 'Unknown')}")
        print(f"   📁 File: {row.get('file_name', 'Unknown')}")
        
        # Check for Judge results
        if 'file_path' in row and pd.notna(row['file_path']):
            file_path = Path(row['file_path'])
            if file_path.exists():
                judge_stats = _load_judge_statistics(file_path)
                if judge_stats:
                    print(f"   ⚖️  Judge Results:")
                    print(f"      • Avg Score: {judge_stats['avg_score']:.2f}/5")
                    print(f"      • Distribution: {judge_stats['distribution']}")
                    print(f"      • Quality: {judge_stats['quality_summary']}")
                else:
                    print(f"   ⚖️  Judge Results: Not available")
            else:
                print(f"   ⚖️  Judge Results: File not found")
        
        print("   " + "-" * 60)
    
    print(f"\n📊 Total Production Runs: {len(registry_df)}")

# ADD this helper function to chatbot.py:
def _load_judge_statistics(file_path):
    """Load and calculate Judge statistics from production result file"""
    try:
        # Try to load Judge_Evaluation sheet
        judge_df = pd.read_excel(file_path, sheet_name='Judge_Evaluation')
        
        # Filter valid scores
        valid_scores = judge_df.dropna(subset=['hallucination_score'])
        
        if len(valid_scores) == 0:
            return None
        
        # Calculate statistics
        scores = valid_scores['hallucination_score']
        avg_score = scores.mean()
        
        # Distribution
        score_counts = scores.value_counts().sort_index()
        distribution = {int(k): int(v) for k, v in score_counts.items()}
        
        # Quality assessment
        excellent_count = len(scores[scores <= 2])
        acceptable_count = len(scores[scores == 3])
        problematic_count = len(scores[scores >= 4])
        
        total = len(scores)
        quality_summary = f"Good: {excellent_count}/{total} ({excellent_count/total*100:.0f}%), Acceptable: {acceptable_count}/{total} ({acceptable_count/total*100:.0f}%), Poor: {problematic_count}/{total} ({problematic_count/total*100:.0f}%)"
        
        return {
            'avg_score': avg_score,
            'distribution': distribution,
            'quality_summary': quality_summary,
            'total_evaluated': total
        }
        
    except Exception as e:
        # Sheet doesn't exist or other error
        return None


def select_prompt_for_production():
    """
    Interaktive Auswahl eines Prompts für die Production Pipeline.
    
    Returns:
        tuple: (prompt_key, prompt_template, prompt_info)
    """
    print(f"\n{'='*60}")
    print("PROMPT AUSWAHL FÜR PRODUCTION RUN")
    print(f"{'='*60}")
    
    available_prompts = list_available_prompts()
    
    print("📝 Verfügbare Prompts:")
    for i, (key, name, description) in enumerate(available_prompts, 1):
        print(f"   {i}. {name}")
        print(f"      → {description}")
        print()
    
    try:
        choice = int(input(f"Wähle Prompt (1-{len(available_prompts)}): ")) - 1
        if not (0 <= choice < len(available_prompts)):
            print("❌ Ungültige Auswahl - verwende Standard (CONCISE)")
            selected_key = 'concise'
        else:
            selected_key = available_prompts[choice][0]
    except ValueError:
        print("❌ Ungültige Eingabe - verwende Standard (CONCISE)")
        selected_key = 'concise'
    
    # Gewählten Prompt laden
    prompt_template = get_prompt_template(selected_key)
    prompt_info = get_prompt_info(selected_key)
    
    print(f"\n✅ Gewählt: {prompt_info['name']}")
    print(f"📝 Beschreibung: {prompt_info['description']}")
    
    # Prompt-Preview anzeigen
    preview = input("\n🔍 Prompt-Preview anzeigen? (y/n): ").lower()
    if preview in ['y', 'yes', 'j', 'ja']:
        print(f"\n{'='*60}")
        print("PROMPT PREVIEW")
        print(f"{'='*60}")
        # Zeige ersten Teil des Prompts
        preview_text = prompt_template[:500] + "..." if len(prompt_template) > 500 else prompt_template
        print(preview_text)
        print(f"{'='*60}")
    
    return selected_key, prompt_template, prompt_info

# MODIFIZIERTE create_production_prompt Funktion
def create_production_prompt(row, prompt_template, prompt_key):
    """
    Erstellt Production Prompt mit dem gewählten Template
    """
    # Für "without_context" wird kein hierarchical_context verwendet
    if prompt_key == 'without_context':
        return prompt_template.format(
            node_label=row['node_label'],
            node_code=row['node_code'],
            table_code=row['table_code']
        )
    else:
        # Standard: mit hierarchical_context
        return prompt_template.format(
            node_label=row['node_label'],
            node_code=row['node_code'],
            table_code=row['table_code'],
            hierarchical_context=row['hierarchical_context']
        )

# MODIFIZIERTE run_production_pipeline Funktion
def run_production_pipeline(nodes_df, config, scope, prompt_key, prompt_template, prompt_info):
    """
    Run production pipeline with enhanced parameters and selected prompt
    """
    
    # Test API connection first
    if not test_api_connection(config):
        print("❌ Cannot proceed without working API connection")
        return None
    
    file_manager = ProductionFileManager(config['output_dir'])
    
    print(f"\n🏭 Starting production run: {scope}")
    print(f"📊 Processing {len(nodes_df)} nodes")
    print(f"📝 Using Prompt: {prompt_info['name']}")
    if config.get('llm_parameters'):
        print(f"🔧 LLM Parameters: {config['llm_parameters']}")
    else:
        print(f"🔧 LLM Parameters: Standard (default)")
    
    results = []
    total_nodes = len(nodes_df)
    start_time = datetime.now()
    
    # Process in batches with progress tracking
    for batch_start in range(0, total_nodes, config['batch_size']):
        batch_end = min(batch_start + config['batch_size'], total_nodes)
        batch_nodes = nodes_df.iloc[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_start+1}-{batch_end} ({len(batch_nodes)} nodes)")
        
        for idx, row in progress_bar(list(batch_nodes.iterrows())):
            # Create production prompt with selected template
            prompt = create_production_prompt(row, prompt_template, prompt_key)
            
            # Generate response with optional parameters
            response_start = datetime.now()
            response = call_chatbot_api(prompt, config)
            processing_time = (datetime.now() - response_start).total_seconds()
            
            # Get LLM parameters for storage
            llm_params = config.get('llm_parameters') or {}
            
            results.append({
                'node_id': f"{row['table_code']}|{row['component_type']}|{row['node_code']}",
                'node_code': row['node_code'],
                'node_label': row['node_label'],
                'table_code': row['table_code'],
                'component_type': row['component_type'],
                'hierarchical_context': row['hierarchical_context'],
                'full_path': row['full_path'],
                'production_prompt': prompt,
                'chatbot_response': response,
                'response_length': len(response),
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'batch_number': (batch_start // config['batch_size']) + 1,
                # Add LLM parameters to each result
                'llm_temperature': llm_params.get('temperature', 'Standard'),
                'llm_max_tokens': llm_params.get('max_tokens', 'Standard'),
                'llm_top_p': llm_params.get('top_p', 'Standard'),
                'llm_presence_penalty': llm_params.get('presence_penalty', 'Standard'),
                'llm_frequency_penalty': llm_params.get('frequency_penalty', 'Standard'),
                # Add prompt information
                'prompt_key': prompt_key,
                'prompt_name': prompt_info['name'],
                'prompt_description': prompt_info['description']
            })
        
        # Intermediate save after each batch
        if results:
            temp_df = pd.DataFrame(results)
            print(f"✅ Batch {batch_start // config['batch_size'] + 1} completed")
    
    # Final save
    total_time = (datetime.now() - start_time).total_seconds()
    results_df = pd.DataFrame(results)
    
    run_config = {
        'run_id': f"production_{scope}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        'scope': scope,
        'scope_description': PRODUCTION_FILTERS[scope]['description'],
        'node_count': len(results_df),
        'taxonomy': 'COREP_3_2',
        'model_name': config['model_name'],
        'batch_size': config['batch_size'],
        'processing_time': f"{total_time:.1f}s total ({total_time/len(results_df):.1f}s per node)",
        'llm_parameters': config.get('llm_parameters'),  # Can be None for Standard
        'prompt_key': prompt_key,
        'prompt_name': prompt_info['name'],
        'prompt_description': prompt_info['description'],
        'notes': f"Production run with {prompt_info['name']} prompt and {'custom' if config.get('llm_parameters') else 'standard'} parameters. Scope: {scope}"
    }
    
    filename = file_manager.save_production_run(results_df, run_config)
    
    print(f"\n🎉 PRODUCTION RUN COMPLETED!")
    print(f"📊 Processed: {len(results_df)} nodes")
    print(f"📝 Prompt: {prompt_info['name']}")
    print(f"⏱️ Total time: {total_time:.1f}s ({total_time/len(results_df):.1f}s per node)")
    if config.get('llm_parameters'):
        print(f"🔧 Used parameters: {config['llm_parameters']}")
    else:
        print(f"🔧 Used parameters: Standard (default)")
    print(f"📁 Results: {filename}")
    
    return results_df

# MODIFIZIERTE process_pickle_tree_with_chatbot Funktion
def process_pickle_tree_with_chatbot(pickle_path=None, taxonomy_code="COREP_3_2"):
    """Main production function with enhanced parameter control and prompt selection"""
    
    # Load tree structure
    if pickle_path is None:
        pickle_path = f"tree_structures/baumstruktur_{taxonomy_code}.pkl"
    
    if not os.path.exists(pickle_path):
        print(f"❌ Pickle file not found: {pickle_path}")
        return None
    
    print(f"Loading tree structure: {pickle_path}")
    all_trees = load_tree_structure(pickle_path)
    print(f"✅ Loaded {len(all_trees)} tree combinations")
    
    # NEUE PROMPT-AUSWAHL
    prompt_key, prompt_template, prompt_info = select_prompt_for_production()
    
    # Show production options
    print(f"\n{'='*60}")
    print("PRODUCTION PIPELINE OPTIONS")
    print(f"{'='*60}")
    
    for key, filter_info in PRODUCTION_FILTERS.items():
        print(f"{key}: {filter_info['description']}")
    
    choice = input(f"\nChoose production scope: ").strip()
    
    if choice not in PRODUCTION_FILTERS:
        print(f"❌ Invalid choice. Available options: {list(PRODUCTION_FILTERS.keys())}")
        return None
    
    # Extract nodes based on selected filter
    filter_config = PRODUCTION_FILTERS[choice]
    print(f"\n🔄 Extracting nodes for: {filter_config['description']}")
    
    nodes_df = extract_nodes_with_filter(all_trees, filter_config)
    print(f"✅ Extracted {len(nodes_df)} nodes")
    
    if len(nodes_df) == 0:
        print("❌ No nodes found with selected filter!")
        return None
    
    # Show preview
    print(f"\n📋 Preview (first 3 nodes):")
    for idx, row in nodes_df.head(3).iterrows():
        print(f"  • {row['node_code']}: {row['node_label']}")
    
    # Show LLM parameters
    print(f"\n🔧 LLM Parameters to be used:")
    if CONFIG.get('llm_parameters'):
        for param, value in CONFIG['llm_parameters'].items():
            print(f"  • {param}: {value}")
    else:
        print(f"  • Standard (default parameters)")
    
    # Confirm production run
    confirm = input(f"\n🚀 Start production run for {len(nodes_df)} nodes with '{prompt_info['name']}' prompt? (y/n): ").lower()
    
    if confirm in ['y', 'yes', 'j', 'ja']:
        return run_production_pipeline(nodes_df, CONFIG, choice, prompt_key, prompt_template, prompt_info)
    else:
        print("❌ Production run cancelled")
        return nodes_df