from database.db_manager import create_tables
from scripts.load_data import load_dpm_to_db, read_excel_data, load_finrep_y_reference, load_tablestructurehierarchy, load_hue_its
from scripts.merge_data import find_correct_membername_for_reference, create_output, create_output_corep, createUpload
from scripts.chatbot import process_pickle_tree_with_chatbot, view_production_results  # Production imports
from grid_search import run_hyperparameter_grid_search
import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime
import re

def main():
    print("Wähle eine Aktion:")
    print("1: Tabellen erstellen/create_tables()")
    print("2: ITS Base Data laden/load_hue_its()")
    print("3: DPM_TableStructure hochladen/Hierachie")
    print("4: 🚀 PRODUCTION: ChatBot-Beschreibungen generieren")
    print("5: 📊 Production Results anzeigen")
    print("6: 🔍 LLM Judge - Benchmark Vergleich")
    print("7: 📤 CREATE UPLOAD: Production Run zu GemKonz")
    print("8: 🔬 GRID SEARCH: Hyperparameter Testing")
    
    #print("23: ?DPM Daten laden")
    #print("25: ?Finrep-Reference Daten laden")
    #print("27: ?Match Merge-Tabelle mit Reference")
    #print("28: ?Finrep - Reference (y-axis) hochladen")
    #print("29: DPM_TableStructure hochladen/Hierachie")
    #print("210: Create Output")
    #print("211: Create Corep-output")
    
    choice = input("Deine Auswahl: ")
    
    if choice == "1":
        create_tables()
        print("Tabellen wurden erstellt.")
        
    elif choice == "2":
        load_hue_its()
        
    elif choice == "3":
        load_tablestructurehierarchy("data/qDPM_TableStructure.xlsx")      
        
    elif choice == "4":
        print("\n" + "="*60)
        print("PRODUCTION CHATBOT PIPELINE")
        print("="*60)
        
        # Show available pickle files
        tree_dir = Path("tree_structures")
        if tree_dir.exists():
            pickle_files = list(tree_dir.glob("*.pkl"))
            if pickle_files:
                print("📁 Available Pickle Files:")
                for i, file in enumerate(pickle_files, 1):
                    print(f"   {i}. {file.name}")
        
        print("\nTaxonomy Options:")
        print("1: COREP_3_2 (recommended)")
        print("2: FINREP_3_2_1")
        print("3: Custom pickle file")
        
        sub_choice = input("\nChoose taxonomy (1-3): ")
        
        if sub_choice == "1":
            print("🔄 Loading COREP_3_2 for production...")
            result_df = process_pickle_tree_with_chatbot(taxonomy_code="COREP_3_2")
            if result_df is not None:
                print(f"✅ Production completed. {len(result_df)} nodes processed.")
                
        elif sub_choice == "2":
            print("🔄 Loading FINREP_3_2_1 for production...")
            result_df = process_pickle_tree_with_chatbot(taxonomy_code="FINREP_3_2_1")
            if result_df is not None:
                print(f"✅ Production completed. {len(result_df)} nodes processed.")
                
        elif sub_choice == "3":
            custom_path = input("Enter path to pickle file: ")
            if Path(custom_path).exists():
                print(f"🔄 Loading {custom_path} for production...")
                result_df = process_pickle_tree_with_chatbot(pickle_path=custom_path)
                if result_df is not None:
                    print(f"✅ Production completed. {len(result_df)} nodes processed.")
            else:
                print(f"❌ File not found: {custom_path}")
        else:
            print("❌ Invalid choice!")
    
    elif choice == "5":
        print("\n" + "="*60)
        print("PRODUCTION RESULTS VIEWER")
        print("="*60)
        view_production_results()
    
    elif choice == "6":
        print("\n" + "="*60)
        print("LLM JUDGE - BENCHMARK VERGLEICH")
        print("="*60)
        run_llm_judge_evaluation()


    elif choice == "7":  # <-- NEUE OPTION HANDLING
        print("\n" + "="*60)
        print("CREATE UPLOAD - PRODUCTION RUN ZU GEMKONZ")
        print("="*60)
        createUpload()

    elif choice == "8":  # <-- NEUE OPTION HANDLING
        print("\n" + "="*60)
        print("HYPERPARAMETER GRID SEARCH")
        print("="*60)
        run_hyperparameter_grid_search()
    
    # Existing choices continue...
    elif choice == "23":
        load_dpm_to_db("data/qDPM_DataPointCategorisations.csv")
        
    elif choice == "25":
        file_path = "finrep_references/finrep_reference_x_axis.xlsx"
        read_excel_data(file_path)
        file_path = "finrep_references/finrep_reference_y_axis.xlsx"
        read_excel_data(file_path)
        
    elif choice == "27":
        find_correct_membername_for_reference()
        
    elif choice == "28":
        load_finrep_y_reference("finrep_references/annex_v_result.csv")
        
    elif choice == "210":
        create_output()
        
    elif choice == "211":
        create_output_corep()
    
    else:
        print("Ungültige Auswahl.")

# Enhanced LLM Judge Configuration with strict parameters
JUDGE_CONFIG = {
    'web_ui_token': 'sk-15b54c10119c45f7a45e790a109d7c8b',
    'model_name': 'chatbot-mistral',
    'web_ui_base_url': 'https://chatbot-open-webui.apps.prod.w.oenb.co.at/',
    
    # Strict Judge-specific LLM parameters
    'llm_parameters': {
        'temperature': 0,          # Maximale Konsistenz für Judge
        'max_tokens': 300,         # Fokussierte Bewertungen
        'top_p': 0.8,              # Präzise Wortwahl
        'presence_penalty': 0.2,   # Vermeidung von Wiederholungen
        'frequency_penalty': 0.2   # Vermeidung von Wort-Wiederholungen
    },
    
    # System parameters
    'timeout': 30,
    'batch_size': 5
}

def parse_node_id(node_id):
    """Parse experiment node_id to extract table, type, and code"""
    # Format: "C 01.00 - Capital Adequacy - Own funds definition|Table row|0010"
    parts = node_id.split('|')
    
    if len(parts) != 3:
        return None, None, None
    
    table_part = parts[0]  # "C 01.00 - Capital Adequacy - Own funds definition"
    type_part = parts[1]   # "Table row"
    code_part = parts[2]   # "0010"
    
    # Extract table code (everything before first " - ")
    table_code = table_part.split(' - ')[0] if ' - ' in table_part else table_part
    
    return table_code, type_part, code_part

def parse_benchmark_table(table_string):
    """Parse benchmark table string to extract table code"""
    # Format: "C 01.00 - OWN FUNDS (CA1)" -> "C 01.00"
    # Handle different separators: - and –
    for separator in [' - ', ' – ', ' — ']:
        if separator in table_string:
            return table_string.split(separator)[0]
    
    return table_string

def load_benchmark_files():
    """Load benchmark files for comparison"""
    benchmark_dir = Path("benchmark_answers")
    
    if not benchmark_dir.exists():
        print(f"❌ Benchmark directory not found: {benchmark_dir}")
        return None, None
    
    rows_file = benchmark_dir / "rows_FULL_cleaned_evaluated.xlsx"
    columns_file = benchmark_dir / "columns_FULL_cleaned_evaluated.xlsx"
    
    rows_df = None
    columns_df = None
    
    try:
        if rows_file.exists():
            rows_df = pd.read_excel(rows_file)
            print(f"✅ Loaded rows benchmark: {len(rows_df)} entries")
        else:
            print(f"⚠️ Rows benchmark not found: {rows_file}")
        
        if columns_file.exists():
            columns_df = pd.read_excel(columns_file)
            print(f"✅ Loaded columns benchmark: {len(columns_df)} entries")
        else:
            print(f"⚠️ Columns benchmark not found: {columns_file}")
            
    except Exception as e:
        print(f"❌ Error loading benchmark files: {e}")
        return None, None
    
    return rows_df, columns_df

def find_benchmark_match(exp_table, exp_type, exp_code, rows_df, columns_df):
    """Find matching benchmark entry for experiment data"""
    
    # Determine which benchmark to use
    if "row" in exp_type.lower():
        benchmark_df = rows_df
        type_column = "Row"
    elif "column" in exp_type.lower():
        benchmark_df = columns_df
        type_column = "Column"
    else:
        return None
    
    if benchmark_df is None:
        return None
    
    # First try primary benchmark
    match = _search_in_benchmark(benchmark_df, type_column, exp_table, exp_code, f"PRIMARY ({type_column})")
    if match is not None:
        return match
    
    # If no match in primary, try the other benchmark as fallback
    if "row" in exp_type.lower() and columns_df is not None:
        fallback_match = _search_in_benchmark(columns_df, "Column", exp_table, exp_code, "FALLBACK (Column)")
        if fallback_match is not None:
            return fallback_match
    
    elif "column" in exp_type.lower() and rows_df is not None:
        fallback_match = _search_in_benchmark(rows_df, "Row", exp_table, exp_code, "FALLBACK (Row)")
        if fallback_match is not None:
            return fallback_match
    
    return None

def normalize_code(code):
    """Normalize codes by removing leading zeros for comparison"""
    code_str = str(code).strip()
    # Remove leading zeros but keep at least one digit
    normalized = code_str.lstrip('0') or '0'
    return normalized

def _search_in_benchmark(benchmark_df, type_column, exp_table, exp_code, search_type):
    """Helper function to search in a specific benchmark DataFrame"""
    
    # Normalize the experiment code for comparison
    exp_code_normalized = normalize_code(exp_code)
    
    # Find matching row - search ALL entries
    for idx, row in benchmark_df.iterrows():
        benchmark_table = parse_benchmark_table(str(row.get('Table', '')))
        benchmark_code = str(row.get(type_column, '')).strip()
        benchmark_code_normalized = normalize_code(benchmark_code)
        
        if (benchmark_table == exp_table and 
            benchmark_code_normalized == exp_code_normalized):
            print(f"    ✅ Match found: {exp_table} {exp_code} -> {search_type}")
            return row
    
    print(f"    ❌ No match in {search_type} for {exp_table} {exp_code}")
    return None

def call_judge_api(prompt, config):
    """Enhanced Judge API call with strict LLM parameters"""
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
    
    # Add strict Judge LLM parameters
    if 'llm_parameters' in config:
        payload.update(config['llm_parameters'])
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=config['timeout'])
        
        if response.status_code != 200:
            return f"API Error: HTTP {response.status_code}"
        
        json_response = response.json()
        
        if "choices" in json_response and json_response["choices"]:
            return json_response["choices"][0]["message"]["content"]
        else:
            return "API Error: Unexpected response format"
        
    except Exception as e:
        return f"Error: {str(e)}"

def create_judge_prompt(chatbot_response, benchmark_data):
    """Create simplified LLM Judge prompt for hallucination detection - 1-5 scale, fully English"""
    
    # Extract benchmark reference texts
    text_ohne_ueberschrift = str(benchmark_data.get('Text ohne Überschrift', '')).strip()
    excerpts = str(benchmark_data.get('Excerpts', '')).strip()
    benchmark_chatbot = str(benchmark_data.get('ChatBot', '')).strip()
    
    prompt = f"""You are an expert evaluator for regulatory content accuracy. Your task is to detect hallucinations in chatbot responses.

CHATBOT RESPONSE TO EVALUATE:
{chatbot_response}

REFERENCE SOURCES (Truth):
1. Official Text: {text_ohne_ueberschrift}
2. Excerpts: {excerpts}
3. Benchmark ChatBot: {benchmark_chatbot}

TASK: Rate the hallucination level using a 1-5 scale:

HALLUCINATION RATING SCALE:
1 = No Hallucinations (all information is backed by reference sources)
2 = Minimal Hallucinations (almost all information is correctly supported, minor deviations)
3 = Some Hallucinations (partially unsupported information, but mostly correct)
4 = Significant Hallucinations (many unsupported claims or invented details)
5 = Severe Hallucinations (mostly invented or unverifiable information)

IMPORTANT: Be strict in your evaluation. Any information that cannot be verified in the reference sources counts as hallucination.

FORMAT YOUR RESPONSE EXACTLY AS:
HALLUCINATION_SCORE: [1-5]
REASONING: [Detailed justification for the score]"""
    
    return prompt

def parse_judge_response(judge_response):
    """Parse simplified LLM Judge response to extract hallucination score (1-5) and reasoning"""
    result = {
        'hallucination_score': None,
        'hallucination_reasoning': ''
    }
    
    lines = judge_response.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('HALLUCINATION_SCORE:'):
            score_part = line.replace('HALLUCINATION_SCORE:', '').strip()
            try:
                score = int(score_part)
                if 1 <= score <= 5:  # Valid 1-5 scale
                    result['hallucination_score'] = score
            except:
                pass
        
        elif line.startswith('REASONING:'):
            reasoning = line.replace('REASONING:', '').strip()
            result['hallucination_reasoning'] = reasoning
    
    return result

def evaluate_experiment_with_judge(experiment_file, rows_df, columns_df):
    """Evaluate experiment responses using enhanced LLM Judge with strict parameters"""
    
    print(f"🔍 Evaluating: {experiment_file.name}")
    print(f"🔧 Judge parameters: {JUDGE_CONFIG.get('llm_parameters', 'default')}")
    
    # Load experiment data
    try:
        exp_df = pd.read_excel(experiment_file, sheet_name='Results')
    except Exception as e:
        print(f"❌ Error loading experiment file: {e}")
        return None
    
    print(f"📊 Found {len(exp_df)} responses to evaluate")
    
    evaluation_results = []
    
    for idx, row in exp_df.iterrows():
        print(f"   Evaluating {idx+1}/{len(exp_df)}: {row['node_code']}")
        
        # Parse node_id
        exp_table, exp_type, exp_code = parse_node_id(row['node_id'])
        
        if not all([exp_table, exp_type, exp_code]):
            print(f"   ⚠️ Could not parse node_id: {row['node_id']}")
            evaluation_results.append({
                'node_id': row['node_id'],
                'node_code': row['node_code'],
                'node_label': row['node_label'],
                'chatbot_response': row['chatbot_response'],
                'benchmark_match': 'Parse Error',
                'hallucination_score': None,
                'hallucination_reasoning': 'Could not parse node_id',
                'judge_response': '',
                'judge_temperature': JUDGE_CONFIG.get('llm_parameters', {}).get('temperature', 'default'),
                'judge_max_tokens': JUDGE_CONFIG.get('llm_parameters', {}).get('max_tokens', 'default'),
                'judge_top_p': JUDGE_CONFIG.get('llm_parameters', {}).get('top_p', 'default')
            })
            continue
        
        # Find benchmark match
        benchmark_match = find_benchmark_match(exp_table, exp_type, exp_code, rows_df, columns_df)
        
        if benchmark_match is None:
            print(f"   ⚠️ No benchmark match found")
            evaluation_results.append({
                'node_id': row['node_id'],
                'node_code': row['node_code'],
                'node_label': row['node_label'],
                'chatbot_response': row['chatbot_response'],
                'benchmark_match': 'No Match',
                'hallucination_score': None,
                'hallucination_reasoning': 'No benchmark match found',
                'judge_response': '',
                'judge_temperature': JUDGE_CONFIG.get('llm_parameters', {}).get('temperature', 'default'),
                'judge_max_tokens': JUDGE_CONFIG.get('llm_parameters', {}).get('max_tokens', 'default'),
                'judge_top_p': JUDGE_CONFIG.get('llm_parameters', {}).get('top_p', 'default')
            })
            continue
        
        # Create judge prompt
        judge_prompt = create_judge_prompt(row['chatbot_response'], benchmark_match)
        
        # Call LLM Judge with enhanced parameters
        judge_response = call_judge_api(judge_prompt, JUDGE_CONFIG)
        
        # Parse judge response
        judge_result = parse_judge_response(judge_response)
        
        evaluation_results.append({
            'node_id': row['node_id'],
            'node_code': row['node_code'],
            'node_label': row['node_label'],
            'chatbot_response': row['chatbot_response'],
            'benchmark_match': 'Found',
            'benchmark_text': str(benchmark_match.get('Text ohne Überschrift', '')),
            'benchmark_excerpts': str(benchmark_match.get('Excerpts', '')),
            'benchmark_chatbot': str(benchmark_match.get('ChatBot', '')),
            'hallucination_score': judge_result['hallucination_score'],
            'hallucination_reasoning': judge_result['hallucination_reasoning'],
            'judge_response': judge_response,
            'judge_temperature': JUDGE_CONFIG.get('llm_parameters', {}).get('temperature', 'default'),
            'judge_max_tokens': JUDGE_CONFIG.get('llm_parameters', {}).get('max_tokens', 'default'),
            'judge_top_p': JUDGE_CONFIG.get('llm_parameters', {}).get('top_p', 'default'),
            'judge_presence_penalty': JUDGE_CONFIG.get('llm_parameters', {}).get('presence_penalty', 'default'),
            'judge_frequency_penalty': JUDGE_CONFIG.get('llm_parameters', {}).get('frequency_penalty', 'default')
        })
        
        # Add small delay to avoid rate limiting
        if idx < len(exp_df) - 1:  # Don't delay after last request
            import time
            time.sleep(1)  # 1 second delay between requests
    
    return pd.DataFrame(evaluation_results)

def save_judge_results_to_experiment(experiment_file, judge_results_df):
    """Add enhanced judge results with multiple metrics to original experiment file"""
    
    try:
        # Read existing Excel file
        with pd.ExcelWriter(experiment_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # Save main judge results
            judge_results_df.to_excel(writer, sheet_name='Enhanced_Judge_Evaluation', index=False)
            
            # Create enhanced statistics summary
            stats_df = create_enhanced_statistics(judge_results_df)
            stats_df.to_excel(writer, sheet_name='Enhanced_Judge_Statistics', index=False)
            
            # Save Judge configuration (wie vorher)
            judge_config_df = create_judge_config_info()
            judge_config_df.to_excel(writer, sheet_name='Judge_Config', index=False)
        
        print(f"✅ Enhanced Judge results added to: {experiment_file}")
        
        # Print enhanced summary
        if not judge_results_df.empty:
            # Count different response types
            informative_count = len(judge_results_df[judge_results_df['response_type'] == 'INFORMATIVE'])
            non_informative_count = len(judge_results_df[judge_results_df['response_type'] == 'NON_INFORMATIVE'])
            
            print(f"📊 Enhanced Summary:")
            print(f"   • Total Responses: {len(judge_results_df)}")
            print(f"   • Informative: {informative_count}")
            print(f"   • Non-Informative: {non_informative_count}")
            
            # Hallucination scores (only informative)
            informative_df = judge_results_df[judge_results_df['response_type'] == 'INFORMATIVE']
            valid_halluc_scores = informative_df.dropna(subset=['hallucination_score'])
            if not valid_halluc_scores.empty:
                avg_halluc = valid_halluc_scores['hallucination_score'].mean()
                print(f"   • Avg Hallucination Score (Informative): {avg_halluc:.2f}")
            
            # Informativeness scores (all responses)
            valid_info_scores = judge_results_df.dropna(subset=['informativeness_score'])
            if not valid_info_scores.empty:
                avg_info = valid_info_scores['informativeness_score'].mean()
                print(f"   • Avg Informativeness Score (All): {avg_info:.2f}")
            
            # Evasiveness rate
            if non_informative_count > 0:
                non_informative_df = judge_results_df[judge_results_df['response_type'] == 'NON_INFORMATIVE']
                valid_justif_scores = non_informative_df.dropna(subset=['justification_score'])
                if not valid_justif_scores.empty:
                    evasive_count = len(valid_justif_scores[valid_justif_scores['justification_score'] <= 2])
                    evasiveness_rate = (evasive_count / non_informative_count) * 100
                    print(f"   • Evasiveness Rate: {evasiveness_rate:.1f}% ({evasive_count}/{non_informative_count})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving enhanced judge results: {e}")
        return False

def create_enhanced_judge_statistics(judge_results_df):
    """Create enhanced statistics for judge results including LLM parameters"""
    
    # Filter valid results
    valid_results = judge_results_df[judge_results_df['benchmark_match'] == 'Found'].copy()
    valid_scores = valid_results.dropna(subset=['hallucination_score'])
    
    if len(valid_scores) == 0:
        return pd.DataFrame([['No valid results found', '']], columns=['Metric', 'Value'])
    
    # Calculate statistics
    stats_data = []
    
    # Basic statistics
    stats_data.append(['Experiment Statistics', ''])
    stats_data.append(['Total Responses', len(judge_results_df)])
    stats_data.append(['Benchmark Matches Found', len(valid_results)])
    stats_data.append(['Valid Scores', len(valid_scores)])
    stats_data.append(['', ''])
    
    # Judge LLM Parameters
    stats_data.append(['Judge LLM Parameters', ''])
    stats_data.append(['Temperature', JUDGE_CONFIG.get('llm_parameters', {}).get('temperature', 'default')])
    stats_data.append(['Max Tokens', JUDGE_CONFIG.get('llm_parameters', {}).get('max_tokens', 'default')])
    stats_data.append(['Top P', JUDGE_CONFIG.get('llm_parameters', {}).get('top_p', 'default')])
    stats_data.append(['Presence Penalty', JUDGE_CONFIG.get('llm_parameters', {}).get('presence_penalty', 'default')])
    stats_data.append(['Frequency Penalty', JUDGE_CONFIG.get('llm_parameters', {}).get('frequency_penalty', 'default')])
    stats_data.append(['', ''])
    
    # Hallucination Score Statistics
    stats_data.append(['Hallucination Score Statistics', ''])
    stats_data.append(['Average Score', f"{valid_scores['hallucination_score'].mean():.2f}"])
    stats_data.append(['Standard Deviation', f"{valid_scores['hallucination_score'].std():.2f}"])
    stats_data.append(['Variance', f"{valid_scores['hallucination_score'].var():.2f}"])
    stats_data.append(['Median', f"{valid_scores['hallucination_score'].median():.1f}"])
    stats_data.append(['Best Score', f"{valid_scores['hallucination_score'].min():.0f}"])
    stats_data.append(['Worst Score', f"{valid_scores['hallucination_score'].max():.0f}"])
    stats_data.append(['', ''])
    
    # Score distribution
    stats_data.append(['Score Distribution', ''])
    score_counts = valid_scores['hallucination_score'].value_counts().sort_index()
    
    score_names = {
        1: 'No Hallucinations',
        2: 'Minimal Hallucinations', 
        3: 'Some Hallucinations',
        4: 'Significant Hallucinations',
        5: 'Severe Hallucinations'
    }
    
    for score in range(1, 6):
        count = score_counts.get(score, 0)
        percentage = (count / len(valid_scores)) * 100 if len(valid_scores) > 0 else 0
        stats_data.append([f'Score {score} ({score_names[score]})', f'{count} ({percentage:.1f}%)'])
    
    stats_data.append(['', ''])
    
    # Quality assessment
    stats_data.append(['Quality Assessment', ''])
    excellent_count = len(valid_scores[valid_scores['hallucination_score'] <= 2])
    acceptable_count = len(valid_scores[valid_scores['hallucination_score'] == 3])
    problematic_count = len(valid_scores[valid_scores['hallucination_score'] >= 4])
    
    stats_data.append(['Excellent (Score 1-2)', f'{excellent_count} ({excellent_count/len(valid_scores)*100:.1f}%)'])
    stats_data.append(['Acceptable (Score 3)', f'{acceptable_count} ({acceptable_count/len(valid_scores)*100:.1f}%)'])
    stats_data.append(['Problematic (Score 4-5)', f'{problematic_count} ({problematic_count/len(valid_scores)*100:.1f}%)'])
    stats_data.append(['', ''])
    
    # Benchmark matching statistics
    stats_data.append(['Benchmark Matching', ''])
    match_counts = judge_results_df['benchmark_match'].value_counts()
    for match_type, count in match_counts.items():
        percentage = (count / len(judge_results_df)) * 100
        stats_data.append([match_type, f'{count} ({percentage:.1f}%)'])
    
    return pd.DataFrame(stats_data, columns=['Metric', 'Value'])

def create_judge_config_info():
    """Create Judge configuration information sheet"""
    config_data = []
    
    config_data.append(['Judge Configuration', ''])
    config_data.append(['Model', JUDGE_CONFIG['model_name']])
    config_data.append(['Base URL', JUDGE_CONFIG['web_ui_base_url']])
    config_data.append(['Timeout', JUDGE_CONFIG['timeout']])
    config_data.append(['Batch Size', JUDGE_CONFIG['batch_size']])
    config_data.append(['', ''])
    
    config_data.append(['LLM Parameters', ''])
    for param, value in JUDGE_CONFIG.get('llm_parameters', {}).items():
        config_data.append([param.title(), value])
    
    config_data.append(['', ''])
    config_data.append(['Evaluation Date', datetime.now().isoformat()])
    config_data.append(['Evaluation Type', 'Hallucination Detection (1-5 scale)'])
    config_data.append(['Evaluation Criteria', 'Strict verification against reference sources'])
    
    return pd.DataFrame(config_data, columns=['Parameter', 'Value'])

def run_llm_judge_evaluation():
    """Main function to run enhanced LLM Judge evaluation"""
    
    print("🔍 LLM JUDGE - BENCHMARK VERGLEICH")
    print("=" * 50)
    
    # Show Judge configuration
    print(f"🔧 Judge Configuration:")
    print(f"   • Model: {JUDGE_CONFIG['model_name']}")
    print(f"   • Temperature: {JUDGE_CONFIG.get('llm_parameters', {}).get('temperature', 'default')}")
    print(f"   • Max Tokens: {JUDGE_CONFIG.get('llm_parameters', {}).get('max_tokens', 'default')}")
    print(f"   • Top P: {JUDGE_CONFIG.get('llm_parameters', {}).get('top_p', 'default')}")
    
    # Find available experiment files
    experiment_dirs = [
        Path("evaluation_experiments/experiments"),
        Path("evaluation_experiments/production_runs")
    ]
    
    experiment_files = []
    for exp_dir in experiment_dirs:
        if exp_dir.exists():
            experiment_files.extend(list(exp_dir.glob("*.xlsx")))
    
    if not experiment_files:
        print("❌ No experiment files found!")
        print("Run some experiments first (option 4)")
        return
    
    print(f"📁 Found {len(experiment_files)} experiment files:")
    for i, file in enumerate(experiment_files, 1):
        print(f"   {i}. {file.name}")
    
    # User selects experiment
    try:
        choice = int(input(f"\nSelect experiment file (1-{len(experiment_files)}): ")) - 1
        if not (0 <= choice < len(experiment_files)):
            print("❌ Invalid selection")
            return
        
        selected_file = experiment_files[choice]
        
    except ValueError:
        print("❌ Please enter a valid number")
        return
    
    # Load benchmark files
    print(f"\n📚 Loading benchmark files...")
    rows_df, columns_df = load_benchmark_files()
    
    if rows_df is None and columns_df is None:
        print("❌ No benchmark files could be loaded!")
        return
    
    # Test Judge API
    print(f"\n🔍 Testing Judge API...")
    test_response = call_judge_api("Test", JUDGE_CONFIG)
    if test_response.startswith("Error") or test_response.startswith("API Error"):
        print(f"❌ Judge API test failed: {test_response}")
        return
    else:
        print("✅ Judge API working with enhanced parameters")
    
    # Run evaluation
    print(f"\n⚖️ Starting enhanced LLM Judge evaluation...")
    judge_results = evaluate_experiment_with_judge(selected_file, rows_df, columns_df)
    
    if judge_results is None:
        print("❌ Evaluation failed")
        return
    
    # Save results
    success = save_judge_results_to_experiment(selected_file, judge_results)
    
    if success:
        print(f"\n🎉 Enhanced LLM Judge evaluation completed!")
        print(f"📁 Results saved in: {selected_file}")
        print(f"📊 Check 'Judge_Evaluation' sheet for detailed results")
        print(f"🔧 Check 'Judge_Config' sheet for parameter details")
    else:
        print("❌ Failed to save results")

def view_production_results():
    """View production run results - simplified for main.py"""
    from scripts.chatbot import view_production_results as view_prod_results
    view_prod_results()


# ========================================
# ENHANCED JUDGE EVALUATION FUNCTIONS
# Füge diese Funktionen am Ende von main.py hinzu (vor if __name__ == "__main__":)
# ========================================

def is_non_informative_response(chatbot_response):
    """Erkennt nicht-informative Antworten die separat behandelt werden sollten"""
    
    response = chatbot_response.strip().lower()
    
    # Häufige "Weiß nicht" Muster
    non_informative_patterns = [
        "i cannot find sufficient information",
        "i cannot find information",
        "insufficient information",
        "no information available",
        "i don't have information",
        "i cannot provide information", 
        "not enough information",
        "unable to find",
        "no relevant information",
        "cannot locate information",
        "information is not available",
        "i do not have access to"
    ]
    
    # Prüfe ob Response hauptsächlich aus "Weiß nicht" besteht
    for pattern in non_informative_patterns:
        if pattern in response and len(response) < 200:  # Kurze Responses
            return True
    
    return False

def create_enhanced_judge_prompt(chatbot_response, benchmark_data):
    """Verbesserter Judge-Prompt der Non-informative Responses separat behandelt"""
    
    # Prüfe zuerst ob es eine non-informative response ist
    if is_non_informative_response(chatbot_response):
        return create_non_informative_judge_prompt(chatbot_response, benchmark_data)
    
    # Extract benchmark reference texts (wie vorher)
    text_ohne_ueberschrift = str(benchmark_data.get('Text ohne Überschrift', '')).strip()
    excerpts = str(benchmark_data.get('Excerpts', '')).strip()
    benchmark_chatbot = str(benchmark_data.get('ChatBot', '')).strip()
    
    prompt = f"""You are an expert evaluator for regulatory content accuracy. Your task is to detect hallucinations in substantive chatbot responses.

CHATBOT RESPONSE TO EVALUATE:
{chatbot_response}

REFERENCE SOURCES (Truth):
1. Official Text: {text_ohne_ueberschrift}
2. Excerpts: {excerpts}
3. Benchmark ChatBot: {benchmark_chatbot}

TASK: Rate the hallucination level and informativeness using 1-5 scales:

HALLUCINATION RATING SCALE:
1 = No Hallucinations (all information is backed by reference sources)
2 = Minimal Hallucinations (almost all information is correctly supported, minor deviations)
3 = Some Hallucinations (partially unsupported information, but mostly correct)
4 = Significant Hallucinations (many unsupported claims or invented details)
5 = Severe Hallucinations (mostly invented or unverifiable information)

INFORMATIVENESS SCALE:
1 = No useful information provided
2 = Minimal useful information
3 = Some useful information
4 = Good amount of useful information  
5 = Comprehensive and highly informative

IMPORTANT: Be strict in your evaluation. Any information that cannot be verified in the reference sources counts as hallucination.

FORMAT YOUR RESPONSE EXACTLY AS:
HALLUCINATION_SCORE: [1-5]
INFORMATIVENESS_SCORE: [1-5]
REASONING: [Detailed justification for both scores]"""
    
    return prompt

def create_non_informative_judge_prompt(chatbot_response, benchmark_data):
    """Spezieller Prompt für non-informative Responses"""
    
    # Extract benchmark reference texts
    text_ohne_ueberschrift = str(benchmark_data.get('Text ohne Überschrift', '')).strip()
    excerpts = str(benchmark_data.get('Excerpts', '')).strip()
    benchmark_chatbot = str(benchmark_data.get('ChatBot', '')).strip()
    
    prompt = f"""You are evaluating a chatbot response that claims insufficient information is available.

CHATBOT RESPONSE:
{chatbot_response}

AVAILABLE REFERENCE SOURCES:
1. Official Text: {text_ohne_ueberschrift}
2. Excerpts: {excerpts}
3. Benchmark ChatBot: {benchmark_chatbot}

TASK: Evaluate if the "insufficient information" claim is justified.

EVALUATION CRITERIA:
- Is there actually sufficient information in the reference sources to answer?
- Is the chatbot appropriately conservative or unnecessarily evasive?

JUSTIFICATION SCALE:
1 = Unjustified evasion (good information was available)
2 = Somewhat unjustified (some information was available)
3 = Neutral (unclear if information was sufficient)
4 = Somewhat justified (limited information available)
5 = Fully justified (truly insufficient information)

FORMAT YOUR RESPONSE EXACTLY AS:
RESPONSE_TYPE: NON_INFORMATIVE
HALLUCINATION_SCORE: N/A
INFORMATIVENESS_SCORE: 1
JUSTIFICATION_SCORE: [1-5]
REASONING: [Why the non-informative response was or wasn't justified]"""
    
    return prompt

def parse_enhanced_judge_response(judge_response):
    """Parse enhanced judge response mit mehreren Scores"""
    
    result = {
        'response_type': 'INFORMATIVE',  # Default
        'hallucination_score': None,
        'informativeness_score': None,
        'justification_score': None,
        'hallucination_reasoning': ''
    }
    
    lines = judge_response.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('RESPONSE_TYPE:'):
            response_type = line.replace('RESPONSE_TYPE:', '').strip()
            result['response_type'] = response_type
            
        elif line.startswith('HALLUCINATION_SCORE:'):
            score_part = line.replace('HALLUCINATION_SCORE:', '').strip()
            if score_part.upper() != 'N/A':
                try:
                    score = int(score_part)
                    if 1 <= score <= 5:
                        result['hallucination_score'] = score
                except:
                    pass
                    
        elif line.startswith('INFORMATIVENESS_SCORE:'):
            score_part = line.replace('INFORMATIVENESS_SCORE:', '').strip()
            try:
                score = int(score_part)
                if 1 <= score <= 5:
                    result['informativeness_score'] = score
            except:
                pass
                
        elif line.startswith('JUSTIFICATION_SCORE:'):
            score_part = line.replace('JUSTIFICATION_SCORE:', '').strip()
            try:
                score = int(score_part)
                if 1 <= score <= 5:
                    result['justification_score'] = score
            except:
                pass
        
        elif line.startswith('REASONING:'):
            reasoning = line.replace('REASONING:', '').strip()
            result['hallucination_reasoning'] = reasoning
    
    return result

def evaluate_experiment_with_enhanced_judge(experiment_file, rows_df, columns_df):
    """Erweiterte Experiment-Evaluierung mit separater Non-Answer Behandlung"""
    
    print(f"🔍 Enhanced Evaluation: {experiment_file.name}")
    print(f"🔧 Judge parameters: {JUDGE_CONFIG.get('llm_parameters', 'default')}")
    
    # Load experiment data
    try:
        exp_df = pd.read_excel(experiment_file, sheet_name='Results')
    except Exception as e:
        print(f"❌ Error loading experiment file: {e}")
        return None
    
    print(f"📊 Found {len(exp_df)} responses to evaluate")
    
    evaluation_results = []
    non_informative_count = 0
    
    for idx, row in exp_df.iterrows():
        print(f"   Evaluating {idx+1}/{len(exp_df)}: {row['node_code']}")
        
        # Parse node_id (wie vorher)
        exp_table, exp_type, exp_code = parse_node_id(row['node_id'])
        
        if not all([exp_table, exp_type, exp_code]):
            print(f"   ⚠️ Could not parse node_id: {row['node_id']}")
            evaluation_results.append({
                'node_id': row['node_id'],
                'node_code': row['node_code'],
                'node_label': row['node_label'],
                'chatbot_response': row['chatbot_response'],
                'response_type': 'PARSE_ERROR',
                'benchmark_match': 'Parse Error',
                'hallucination_score': None,
                'informativeness_score': None,
                'justification_score': None,
                'hallucination_reasoning': 'Could not parse node_id',
                'judge_response': '',
            })
            continue
        
        # Find benchmark match (wie vorher)
        benchmark_match = find_benchmark_match(exp_table, exp_type, exp_code, rows_df, columns_df)
        
        if benchmark_match is None:
            print(f"   ⚠️ No benchmark match found")
            evaluation_results.append({
                'node_id': row['node_id'],
                'node_code': row['node_code'],
                'node_label': row['node_label'],
                'chatbot_response': row['chatbot_response'],
                'response_type': 'NO_BENCHMARK',
                'benchmark_match': 'No Match',
                'hallucination_score': None,
                'informativeness_score': None,
                'justification_score': None,
                'hallucination_reasoning': 'No benchmark match found',
                'judge_response': '',
            })
            continue
        
        # Check if response is non-informative
        is_non_informative = is_non_informative_response(row['chatbot_response'])
        if is_non_informative:
            non_informative_count += 1
            print(f"   📝 Non-informative response detected")
        
        # Create enhanced judge prompt
        judge_prompt = create_enhanced_judge_prompt(row['chatbot_response'], benchmark_match)
        
        # Call LLM Judge
        judge_response = call_judge_api(judge_prompt, JUDGE_CONFIG)
        
        # Parse enhanced judge response
        judge_result = parse_enhanced_judge_response(judge_response)
        
        evaluation_results.append({
            'node_id': row['node_id'],
            'node_code': row['node_code'],
            'node_label': row['node_label'],
            'chatbot_response': row['chatbot_response'],
            'response_type': judge_result['response_type'],
            'benchmark_match': 'Found',
            'benchmark_text': str(benchmark_match.get('Text ohne Überschrift', '')),
            'benchmark_excerpts': str(benchmark_match.get('Excerpts', '')),
            'benchmark_chatbot': str(benchmark_match.get('ChatBot', '')),
            'hallucination_score': judge_result['hallucination_score'],
            'informativeness_score': judge_result['informativeness_score'],
            'justification_score': judge_result['justification_score'],
            'hallucination_reasoning': judge_result['hallucination_reasoning'],
            'judge_response': judge_response,
            'is_non_informative': is_non_informative,
        })
        
        # Add small delay
        if idx < len(exp_df) - 1:
            import time
            time.sleep(1)
    
    result_df = pd.DataFrame(evaluation_results)
    
    print(f"\n📊 ENHANCED EVALUATION SUMMARY:")
    print(f"   Total responses: {len(result_df)}")
    print(f"   Non-informative responses: {non_informative_count}")
    print(f"   Substantive responses: {len(result_df) - non_informative_count}")
    
    return result_df

def create_enhanced_statistics(judge_results_df):
    """Erstelle erweiterte Statistiken die Non-informative Responses berücksichtigen"""
    
    # Separate informative and non-informative responses
    informative_df = judge_results_df[judge_results_df['response_type'] == 'INFORMATIVE'].copy()
    non_informative_df = judge_results_df[judge_results_df['response_type'] == 'NON_INFORMATIVE'].copy()
    
    stats_data = []
    
    # Basic statistics
    stats_data.append(['Enhanced Evaluation Summary', ''])
    stats_data.append(['Total Responses', len(judge_results_df)])
    stats_data.append(['Informative Responses', len(informative_df)])
    stats_data.append(['Non-Informative Responses', len(non_informative_df)])
    stats_data.append(['Non-Informative Rate', f"{len(non_informative_df)/len(judge_results_df)*100:.1f}%"])
    stats_data.append(['', ''])
    
    # Hallucination statistics (only for informative responses)
    if len(informative_df) > 0:
        valid_halluc_scores = informative_df.dropna(subset=['hallucination_score'])
        if len(valid_halluc_scores) > 0:
            stats_data.append(['Hallucination Statistics (Informative Only)', ''])
            stats_data.append(['Average Hallucination Score', f"{valid_halluc_scores['hallucination_score'].mean():.2f}"])
            stats_data.append(['Best Hallucination Score', f"{valid_halluc_scores['hallucination_score'].min():.0f}"])
            stats_data.append(['Worst Hallucination Score', f"{valid_halluc_scores['hallucination_score'].max():.0f}"])
            stats_data.append(['', ''])
    
    # Informativeness statistics (all responses)
    all_informative_scores = judge_results_df.dropna(subset=['informativeness_score'])
    if len(all_informative_scores) > 0:
        stats_data.append(['Informativeness Statistics (All Responses)', ''])
        stats_data.append(['Average Informativeness Score', f"{all_informative_scores['informativeness_score'].mean():.2f}"])
        stats_data.append(['Best Informativeness Score', f"{all_informative_scores['informativeness_score'].max():.0f}"])
        stats_data.append(['Worst Informativeness Score', f"{all_informative_scores['informativeness_score'].min():.0f}"])
        stats_data.append(['', ''])
    
    # Justification statistics (non-informative responses)
    if len(non_informative_df) > 0:
        valid_justif_scores = non_informative_df.dropna(subset=['justification_score'])
        if len(valid_justif_scores) > 0:
            stats_data.append(['Justification Statistics (Non-Informative)', ''])
            stats_data.append(['Average Justification Score', f"{valid_justif_scores['justification_score'].mean():.2f}"])
            stats_data.append(['Well-Justified (Score 4-5)', f"{len(valid_justif_scores[valid_justif_scores['justification_score'] >= 4])}"])
            stats_data.append(['Poorly-Justified (Score 1-2)', f"{len(valid_justif_scores[valid_justif_scores['justification_score'] <= 2])}"])
            stats_data.append(['', ''])
    
    # Quality assessment
    stats_data.append(['Overall Quality Assessment', ''])
    
    # Excellent responses: informative + low hallucination
    if len(informative_df) > 0:
        excellent_responses = len(informative_df[
            (informative_df['hallucination_score'] <= 2) & 
            (informative_df['informativeness_score'] >= 4)
        ])
        stats_data.append(['Excellent (Informative + Low Hallucination)', f"{excellent_responses}"])
    
    # Problematic responses: informative + high hallucination
    if len(informative_df) > 0:
        problematic_responses = len(informative_df[informative_df['hallucination_score'] >= 4])
        stats_data.append(['Problematic (High Hallucination)', f"{problematic_responses}"])
    
    # Evasive responses: unjustified non-informative
    if len(non_informative_df) > 0:
        evasive_responses = len(non_informative_df[non_informative_df['justification_score'] <= 2])
        stats_data.append(['Evasive (Unjustified Non-Informative)', f"{evasive_responses}"])
    
    return pd.DataFrame(stats_data, columns=['Metric', 'Value'])

if __name__ == "__main__":
    main()