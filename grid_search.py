"""
grid_search.py - Hyperparameter Testing Grid-Search System
"""

import itertools
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path
from prompts import list_available_prompts, get_prompt_template, get_prompt_info
from scripts.chatbot import (
    load_tree_structure, extract_nodes_with_filter, PRODUCTION_FILTERS,
    run_production_pipeline, CONFIG
)

# Grid-Search Konfiguration für Hyperparameter-Optimierung
GRID_SEARCH_CONFIG = {
    # Sehr feine Parameter-Abstufungen um die optimalen Werte
    'llm_parameter_sets': {
        
        # === BASELINE (aktuell beste Parameter) ===
        'baseline_optimal': {
            'temperature': 0.3,
            'max_tokens': 150,
            'top_p': 0.8,
            'presence_penalty': 0.05,
            'frequency_penalty': 0.05
        },
        
        # === TEMPERATURE FINE-TUNING  ===
        'temp_0_20': {
            'temperature': 0.0,     
            'max_tokens': 150,
            'top_p': 0.8,
            'presence_penalty': 0.05,
            'frequency_penalty': 0.05
        }
    },
    
    # Welche Prompts testen (None = alle verfügbaren)
    #'test_prompts': None,  # oder ['with_context', 'without_context']
    'test_prompts': ['examples_based'], 
    
    # Production Scopes für Testing
    #'test_scopes': ['first_20'],  # Kleine Samples für schnelle Tests c01_both
    'test_scopes': ['c01_both'],
    
    # Judge Configuration
    'auto_judge': True,  # Automatisch LLM Judge nach jedem Run
    'judge_timeout': 45,  # Längere Timeouts für Judge
    
    # Output
    'results_dir': 'evaluation_experiments/grid_search_results',
    'summary_file': 'grid_search_summary.xlsx'
}

class GridSearchManager:
    """Manager für systematische Hyperparameter-Tests"""
    
    def __init__(self, config=None):
        self.config = config or GRID_SEARCH_CONFIG
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = []
        self.results_summary = []
        
    def generate_experiment_grid(self):
        """Generiert alle Experiment-Kombinationen"""
        
        # Verfügbare Prompts ermitteln
        available_prompts = list_available_prompts()
        test_prompts = self.config.get('test_prompts')
        if test_prompts is None:
            prompt_keys = [key for key, _, _ in available_prompts]
        else:
            prompt_keys = test_prompts
        
        # Prompt-Namen-Mapping erstellen
        prompt_names = {key: name for key, name, _ in available_prompts}
        
        # Parameter-Sets
        param_sets = list(self.config['llm_parameter_sets'].keys())
        
        # Scopes
        test_scopes = self.config['test_scopes']
        
        # Grid erstellen
        experiments = []
        experiment_id = 1
        
        for prompt_key, param_set, scope in itertools.product(prompt_keys, param_sets, test_scopes):
            experiment = {
                'experiment_id': f"EXP_{experiment_id:03d}",
                'prompt_key': prompt_key,
                'prompt_name': prompt_names.get(prompt_key, 'Unknown'),
                'param_set_name': param_set,
                'llm_parameters': self.config['llm_parameter_sets'][param_set],
                'scope': scope,
                'status': 'pending',
                'production_file': None,
                'judge_completed': False,
                'avg_hallucination_score': None,
                'success_rate': None,
                'avg_response_length': None,
                'processing_time': None
            }
            experiments.append(experiment)
            experiment_id += 1
        
        self.experiments = experiments
        print(f"✅ Grid erstellt: {len(experiments)} Experimente")
        print(f"   📝 Prompts: {len(prompt_keys)} ({', '.join(prompt_keys)})")
        print(f"   🔧 Parameter-Sets: {len(param_sets)} ({', '.join(param_sets)})")
        print(f"   🎯 Scopes: {len(test_scopes)} ({', '.join(test_scopes)})")
        
        return experiments

    def run_single_experiment(self, experiment):
        """Führt ein einzelnes Experiment aus - KORRIGIERTE VERSION"""
        
        exp_id = experiment['experiment_id']
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT {exp_id}")
        print(f"📝 Prompt: {experiment['prompt_name']}")
        print(f"🔧 Parameters: {experiment['param_set_name']}")
        print(f"🎯 Scope: {experiment['scope']}")
        print(f"{'='*60}")
        
        try:
            # 1. Tree Structure laden
            tree_path = "tree_structures/baumstruktur_COREP_3_2.pkl"
            if not os.path.exists(tree_path):
                raise FileNotFoundError(f"Tree structure not found: {tree_path}")
            
            all_trees = load_tree_structure(tree_path)
            
            # 2. Nodes extrahieren (mit reduzierter Ausgabe)
            filter_config = PRODUCTION_FILTERS[experiment['scope']]  # VERWENDE ORIGINAL SCOPE
            print(f"   📊 Extrahiere Nodes für Scope: {experiment['scope']}")
            
            # Nutze verbose=False für saubere Grid Search Ausgabe
            nodes_df = extract_nodes_with_filter(all_trees, filter_config, verbose=False)
            
            if len(nodes_df) == 0:
                raise ValueError("No nodes extracted")
            
            print(f"   ✅ {len(nodes_df)} Nodes extrahiert")
            
            # 3. Prompt-Template laden
            prompt_template = get_prompt_template(experiment['prompt_key'])
            prompt_info = get_prompt_info(experiment['prompt_key'])
            
            if not prompt_template or not prompt_info:
                raise ValueError(f"Invalid prompt_key: {experiment['prompt_key']}")
            
            # 4. CONFIG für dieses Experiment anpassen
            experiment_config = CONFIG.copy()
            experiment_config['llm_parameters'] = experiment['llm_parameters']
            
            # 5. Production Run - KORRIGIERT
            start_time = time.time()
            
            # VERWENDE ORIGINAL SCOPE für run_production_pipeline
            # Aber erstelle custom run_config für Dateinamen
            original_scope = experiment['scope']  # z.B. "first_20"
            
            # Temporär den run_production_pipeline-Code hier replizieren, 
            # aber mit angepasster Dateinamen-Logik
            results_df = self._run_grid_search_production(
                nodes_df=nodes_df,
                config=experiment_config,
                scope=original_scope,
                experiment_id=exp_id,
                prompt_key=experiment['prompt_key'],
                prompt_template=prompt_template,
                prompt_info=prompt_info
            )
            
            processing_time = time.time() - start_time
            
            if results_df is None or len(results_df) == 0:
                raise ValueError("Production run failed")
            
            # 6. Production File finden - KORRIGIERTE SUCHE
            production_files = list(Path("evaluation_experiments/production_runs").glob(f"*{exp_id}*.xlsx"))
            if not production_files:
                # Fallback: Suche nach neuesten Files
                production_files = sorted(
                    Path("evaluation_experiments/production_runs").glob("*.xlsx"),
                    key=os.path.getmtime,
                    reverse=True
                )[:3]  # Neueste 3 Files
                
                if production_files:
                    production_file = production_files[0]  # Neueste Datei nehmen
                    print(f"   📁 Verwendete neueste Datei: {production_file.name}")
                else:
                    raise FileNotFoundError("No production files found")
            else:
                production_file = production_files[0]
            
            # 7. Basis-Statistiken berechnen
            successful_responses = len(results_df[~results_df['chatbot_response'].str.startswith(('API Error', 'Error'))])
            success_rate = (successful_responses / len(results_df)) * 100
            avg_response_length = results_df['response_length'].mean()
            
            # 8. Experiment-Status aktualisieren
            experiment.update({
                'status': 'production_completed',
                'production_file': str(production_file),
                'success_rate': success_rate,
                'avg_response_length': avg_response_length,
                'processing_time': processing_time,
                'nodes_processed': len(results_df)
            })
            
            print(f"✅ Production completed: {len(results_df)} nodes, {success_rate:.1f}% success")
            
            # 9. Automatischer LLM Judge (falls aktiviert)
            if self.config.get('auto_judge', False):
                judge_result = self.run_judge_for_experiment(experiment)
                if judge_result:
                    experiment.update(judge_result)
            
            return experiment
            
        except Exception as e:
            print(f"❌ Experiment {exp_id} failed: {str(e)}")
            experiment.update({
                'status': 'failed',
                'error': str(e)
            })
            return experiment

    def _run_grid_search_production(self, nodes_df, config, scope, experiment_id, prompt_key, prompt_template, prompt_info):
        """Spezialisierte Production Pipeline für Grid Search"""
        
        # Import der benötigten Funktionen
        from scripts.chatbot import (
            test_api_connection, ProductionFileManager, call_chatbot_api,
            create_production_prompt
        )
        from fastprogress.fastprogress import progress_bar
        
        # Test API connection first
        if not test_api_connection(config):
            print("❌ Cannot proceed without working API connection")
            return None
        
        file_manager = ProductionFileManager(config['output_dir'])
        
        print(f"🏭 Starting Grid Search production: {experiment_id}")
        print(f"📊 Processing {len(nodes_df)} nodes")
        
        results = []
        total_nodes = len(nodes_df)
        start_time = datetime.now()
        
        # Process in batches with progress tracking
        for batch_start in range(0, total_nodes, config['batch_size']):
            batch_end = min(batch_start + config['batch_size'], total_nodes)
            batch_nodes = nodes_df.iloc[batch_start:batch_end]
            
            print(f"📦 Processing batch {batch_start+1}-{batch_end}")
            
            for idx, row in progress_bar(list(batch_nodes.iterrows())):
                # Create production prompt with selected template
                prompt = create_production_prompt(row, prompt_template, prompt_key)
                
                # Generate response
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
                    'llm_temperature': llm_params.get('temperature', 'Standard'),
                    'llm_max_tokens': llm_params.get('max_tokens', 'Standard'),
                    'llm_top_p': llm_params.get('top_p', 'Standard'),
                    'llm_presence_penalty': llm_params.get('presence_penalty', 'Standard'),
                    'llm_frequency_penalty': llm_params.get('frequency_penalty', 'Standard'),
                    'prompt_key': prompt_key,
                    'prompt_name': prompt_info['name'],
                    'prompt_description': prompt_info['description'],
                    'experiment_id': experiment_id  # Grid Search specific
                })
        
        # Final save
        total_time = (datetime.now() - start_time).total_seconds()
        results_df = pd.DataFrame(results)
        
        # Angepasster scope für Grid Search Dateinamen
        grid_search_scope = f"{scope}_gridSearch_{experiment_id}"
        
        run_config = {
            'run_id': f"gridSearch_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            'scope': grid_search_scope,  # Hier verwenden wir den angepassten Scope
            'scope_description': f"Grid Search Experiment {experiment_id} - {PRODUCTION_FILTERS[scope]['description']}",
            'node_count': len(results_df),
            'taxonomy': 'COREP_3_2',
            'model_name': config['model_name'],
            'batch_size': config['batch_size'],
            'processing_time': f"{total_time:.1f}s total ({total_time/len(results_df):.1f}s per node)",
            'llm_parameters': config.get('llm_parameters'),
            'prompt_key': prompt_key,
            'prompt_name': prompt_info['name'],
            'prompt_description': prompt_info['description'],
            'notes': f"Grid Search Experiment {experiment_id} with {prompt_info['name']} prompt"
        }
        
        filename = file_manager.save_production_run(results_df, run_config)
        
        print(f"✅ Grid Search production completed: {filename}")
        
        return results_df
        
# ERSETZE in grid_search.py die run_judge_for_experiment Methode mit dieser:

    def run_judge_for_experiment(self, experiment):
        """Führt Enhanced LLM Judge für ein Experiment aus"""
        
        exp_id = experiment['experiment_id']
        production_file = experiment.get('production_file')
        
        if not production_file or not os.path.exists(production_file):
            print(f"⚠️ No production file for Judge: {exp_id}")
            return None
        
        try:
            print(f"⚖️ Running Enhanced LLM Judge for {exp_id}...")
            
            # Import Enhanced Judge functions (aus main.py)
            from main import (
                load_benchmark_files, evaluate_experiment_with_enhanced_judge, 
                save_judge_results_to_experiment
            )
            
            # Benchmark Files laden
            rows_df, columns_df = load_benchmark_files()
            if rows_df is None and columns_df is None:
                print(f"⚠️ No benchmark files available for Judge")
                return None
            
            # Enhanced Judge ausführen
            judge_results = evaluate_experiment_with_enhanced_judge(
                Path(production_file), rows_df, columns_df
            )
            
            if judge_results is None or len(judge_results) == 0:
                print(f"⚠️ Enhanced Judge failed for {exp_id}")
                return None
            
            # Enhanced Judge Results speichern
            save_judge_results_to_experiment(Path(production_file), judge_results)
            
            # Enhanced Statistics berechnen
            stats = self._calculate_enhanced_judge_stats(judge_results)
            
            print(f"✅ Enhanced Judge completed for {exp_id}:")
            print(f"   🎯 Hallucination Score: {stats.get('avg_hallucination_score', 'N/A')}")
            print(f"   📈 Informativeness Score: {stats.get('avg_informativeness_score', 'N/A')}")
            print(f"   ⚖️ Evasiveness Rate: {stats.get('evasiveness_rate', 'N/A')}%")
            
            return stats
                
        except Exception as e:
            print(f"❌ Enhanced Judge failed for {exp_id}: {str(e)}")
            return {'judge_completed': False, 'judge_error': str(e), 'status': 'judge_failed'}
    
    def _calculate_enhanced_judge_stats(self, judge_results):
        """Berechnet Enhanced Judge Statistiken für Grid Search"""
        
        if judge_results is None or len(judge_results) == 0:
            return {'judge_completed': False}
        
        # Separate response types
        informative_df = judge_results[judge_results['response_type'] == 'INFORMATIVE']
        non_informative_df = judge_results[judge_results['response_type'] == 'NON_INFORMATIVE']
        
        stats = {
            'judge_completed': True,
            'status': 'completed',
            'total_responses': len(judge_results),
            'informative_responses': len(informative_df),
            'non_informative_responses': len(non_informative_df),
        }
        
        # Hallucination Score (nur informative responses)
        if len(informative_df) > 0:
            valid_halluc_scores = informative_df.dropna(subset=['hallucination_score'])
            if len(valid_halluc_scores) > 0:
                stats['avg_hallucination_score'] = round(valid_halluc_scores['hallucination_score'].mean(), 2)
                stats['hallucination_evaluations'] = len(valid_halluc_scores)
            else:
                stats['avg_hallucination_score'] = None
        else:
            stats['avg_hallucination_score'] = None
        
        # Informativeness Score (alle responses)
        valid_info_scores = judge_results.dropna(subset=['informativeness_score'])
        if len(valid_info_scores) > 0:
            stats['avg_informativeness_score'] = round(valid_info_scores['informativeness_score'].mean(), 2)
            stats['informativeness_evaluations'] = len(valid_info_scores)
        else:
            stats['avg_informativeness_score'] = None
        
        # Evasiveness Rate (non-informative responses)
        if len(non_informative_df) > 0:
            valid_justif_scores = non_informative_df.dropna(subset=['justification_score'])
            if len(valid_justif_scores) > 0:
                evasive_count = len(valid_justif_scores[valid_justif_scores['justification_score'] <= 2])
                stats['evasiveness_rate'] = round((evasive_count / len(non_informative_df)) * 100, 1)
                stats['evasive_responses'] = evasive_count
            else:
                stats['evasiveness_rate'] = None
        else:
            stats['evasiveness_rate'] = 0.0  # Keine non-informative responses = keine evasiveness
        
        # Non-informative rate
        stats['non_informative_rate'] = round((len(non_informative_df) / len(judge_results)) * 100, 1)
        
        return stats
    
    def run_grid_search(self):
        """Führt kompletten Grid-Search aus"""
        
        print(f"\n🚀 STARTING GRID SEARCH")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Grid generieren
        experiments = self.generate_experiment_grid()
        
        # Confirmation
        estimated_time = len(experiments) * 2  # ca. 2 Minuten pro Experiment
        print(f"\n⏱️ Geschätzte Zeit: ~{estimated_time} Minuten")
        confirm = input(f"🚀 Start Grid Search mit {len(experiments)} Experimenten? (y/n): ").lower()
        
        if confirm not in ['y', 'yes', 'j', 'ja']:
            print("❌ Grid Search abgebrochen")
            return None
        
        # Experimente ausführen
        completed_experiments = []
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n📊 Progress: {i}/{len(experiments)}")
            
            result = self.run_single_experiment(experiment)
            completed_experiments.append(result)
            
            # Zwischenspeicherung
            if i % 3 == 0:  # Alle 3 Experimente speichern
                self.save_intermediate_results(completed_experiments)
        
        # Finale Auswertung
        self.create_final_summary(completed_experiments)
        
        print(f"\n🎉 GRID SEARCH COMPLETED!")
        print(f"📊 {len(completed_experiments)} Experimente abgeschlossen")
        
        return completed_experiments
    
    def save_intermediate_results(self, experiments):
        """Speichert Zwischenergebnisse"""
        
        df = pd.DataFrame(experiments)
        interim_file = self.results_dir / f"interim_results_{datetime.now().strftime('%H%M')}.xlsx"
        df.to_excel(interim_file, index=False)
        print(f"💾 Zwischenergebnisse gespeichert: {interim_file.name}")
    
# ERSETZE in grid_search.py die create_final_summary Methode mit dieser:

    def create_final_summary(self, experiments):
        """Erstellt finale Zusammenfassung mit Enhanced Judge Metriken"""
        
        # DataFrame erstellen
        df = pd.DataFrame(experiments)
        
        # Sortieren nach Hallucination Score (aufsteigend = besser), dann nach Informativeness (absteigend = besser)
        df_sorted = df.sort_values(['avg_hallucination_score', 'avg_informativeness_score'], 
                                   ascending=[True, False], na_position='last')
        
        # Zusammenfassung erstellen
        summary_file = self.results_dir / self.config['summary_file']
        
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            # Hauptergebnisse
            df_sorted.to_excel(writer, sheet_name='All_Experiments', index=False)
            
            # Top Performers (nur completed experiments)
            completed_df = df_sorted[df_sorted['status'] == 'completed'].copy()
            
            # Enhanced Statistiken erstellen
            stats_data = []
            stats_data.append(['Enhanced Grid Search Summary', ''])
            stats_data.append(['Total Experiments', len(experiments)])
            stats_data.append(['Completed Successfully', len(completed_df)])
            stats_data.append(['Failed', len(df) - len(completed_df)])
            stats_data.append(['', ''])
            
            if len(completed_df) > 0:
                # Top 5 nur wenn completed experiments vorhanden
                top_5 = completed_df.head(5)
                top_5.to_excel(writer, sheet_name='Top_5_Best', index=False)
                
                # Enhanced Detaillierte Statistiken
                valid_halluc = completed_df.dropna(subset=['avg_hallucination_score'])
                valid_info = completed_df.dropna(subset=['avg_informativeness_score'])
                valid_evasive = completed_df.dropna(subset=['evasiveness_rate'])
                
                if len(valid_halluc) > 0:
                    stats_data.append(['=== HALLUCINATION QUALITY ===', ''])
                    stats_data.append(['Best Avg Hallucination Score', f"{valid_halluc['avg_hallucination_score'].min():.2f}"])
                    stats_data.append(['Worst Avg Hallucination Score', f"{valid_halluc['avg_hallucination_score'].max():.2f}"])
                    stats_data.append(['Overall Avg Hallucination Score', f"{valid_halluc['avg_hallucination_score'].mean():.2f}"])
                    stats_data.append(['', ''])
                
                if len(valid_info) > 0:
                    stats_data.append(['=== INFORMATIVENESS ===', ''])
                    stats_data.append(['Best Avg Informativeness Score', f"{valid_info['avg_informativeness_score'].max():.2f}"])
                    stats_data.append(['Worst Avg Informativeness Score', f"{valid_info['avg_informativeness_score'].min():.2f}"])
                    stats_data.append(['Overall Avg Informativeness Score', f"{valid_info['avg_informativeness_score'].mean():.2f}"])
                    stats_data.append(['', ''])
                
                if len(valid_evasive) > 0:
                    stats_data.append(['=== EVASIVENESS ===', ''])
                    stats_data.append(['Lowest Evasiveness Rate', f"{valid_evasive['evasiveness_rate'].min():.1f}%"])
                    stats_data.append(['Highest Evasiveness Rate', f"{valid_evasive['evasiveness_rate'].max():.1f}%"])
                    stats_data.append(['Overall Avg Evasiveness Rate', f"{valid_evasive['evasiveness_rate'].mean():.1f}%"])
                    stats_data.append(['', ''])
                
                # Best Overall Combination
                if len(valid_halluc) > 0:
                    best_exp = valid_halluc.iloc[0]
                    stats_data.append(['=== BEST OVERALL COMBINATION ===', ''])
                    stats_data.append(['  Experiment ID', best_exp['experiment_id']])
                    stats_data.append(['  Prompt', best_exp['prompt_name']])
                    stats_data.append(['  Parameters', best_exp['param_set_name']])
                    stats_data.append(['  Hallucination Score', f"{best_exp['avg_hallucination_score']:.2f}"])
                    stats_data.append(['  Informativeness Score', f"{best_exp.get('avg_informativeness_score', 'N/A')}"])
                    stats_data.append(['  Evasiveness Rate', f"{best_exp.get('evasiveness_rate', 'N/A')}%"])
                    stats_data.append(['', ''])
                
                # Quality Categories
                stats_data.append(['=== QUALITY CATEGORIES ===', ''])
                
                # Excellent: Low hallucination + High informativeness
                excellent = completed_df[
                    (completed_df['avg_hallucination_score'] <= 2.0) & 
                    (completed_df['avg_informativeness_score'] >= 4.0)
                ]
                stats_data.append(['Excellent (Low Halluc. + High Info.)', f"{len(excellent)}"])
                
                # Good: Low hallucination OR High informativeness
                good = completed_df[
                    ((completed_df['avg_hallucination_score'] <= 2.5) | 
                     (completed_df['avg_informativeness_score'] >= 3.5)) &
                    (~completed_df.index.isin(excellent.index))
                ]
                stats_data.append(['Good (Low Halluc. OR High Info.)', f"{len(good)}"])
                
                # Problematic: High hallucination
                problematic = completed_df[completed_df['avg_hallucination_score'] >= 4.0]
                stats_data.append(['Problematic (High Hallucination)', f"{len(problematic)}"])
                
                # Evasive: High evasiveness rate
                evasive = completed_df[completed_df['evasiveness_rate'] >= 50.0]
                stats_data.append(['Evasive (High Evasiveness Rate)', f"{len(evasive)}"])
                
            else:
                stats_data.append(['No experiments completed successfully', ''])
                # Fehler-Analyse hinzufügen
                failed_df = df[df['status'] == 'failed']
                if len(failed_df) > 0:
                    stats_data.append(['', ''])
                    stats_data.append(['Failed Experiments Analysis', ''])
                    for _, exp in failed_df.iterrows():
                        error_msg = exp.get('error', 'Unknown error')[:100] + "..." if len(str(exp.get('error', ''))) > 100 else exp.get('error', 'Unknown error')
                        stats_data.append([f"  {exp['experiment_id']}", error_msg])
            
            stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Enhanced_Summary_Stats', index=False)
        
        print(f"📊 Enhanced Grid Search summary created: {summary_file}")
        
        # Enhanced Console Summary
        if len(completed_df) > 0:
            print(f"\n🏆 ENHANCED BEST RESULTS:")
            best = completed_df.iloc[0]
            print(f"🥇 Best Overall: {best['experiment_id']}")
            print(f"   Prompt: {best['prompt_name']}")
            print(f"   Parameters: {best['param_set_name']}")
            print(f"   🎯 Hallucination Score: {best.get('avg_hallucination_score', 'N/A')}")
            print(f"   📈 Informativeness Score: {best.get('avg_informativeness_score', 'N/A')}")
            print(f"   ⚖️ Evasiveness Rate: {best.get('evasiveness_rate', 'N/A')}%")
            
            if len(completed_df) >= 3:
                second = completed_df.iloc[1]
                third = completed_df.iloc[2]
                print(f"\n🥈 Second: {second['experiment_id']} - Halluc: {second.get('avg_hallucination_score', 'N/A')}, Info: {second.get('avg_informativeness_score', 'N/A')}")
                print(f"🥉 Third: {third['experiment_id']} - Halluc: {third.get('avg_hallucination_score', 'N/A')}, Info: {third.get('avg_informativeness_score', 'N/A')}")
        
        print(f"\n💡 TIP: Check the 'Enhanced_Summary_Stats' sheet for detailed quality analysis!")
        print(f"📁 Results saved in: {summary_file}")


# Hauptfunktion für main.py Integration
def run_hyperparameter_grid_search():
    """Hauptfunktion für Grid Search - aufrufbar aus main.py"""
    
    print("🔬 HYPERPARAMETER GRID SEARCH")
    print("=" * 60)
    print("Systematische Tests verschiedener Prompt- und Parameter-Kombinationen")
    print("Automatische LLM Judge Bewertung für jedes Experiment")
    print("=" * 60)
    
    # Optional: Custom Configuration
    custom_config = input("\nCustom Config verwenden? (y/n): ").lower()
    
    if custom_config in ['y', 'yes', 'j', 'ja']:
        # Hier könnte man interaktive Config-Anpassung einbauen
        print("📝 Verwende Standard-Config (Erweiterung möglich)")
    
    # Grid Search Manager erstellen und ausführen
    manager = GridSearchManager()
    results = manager.run_grid_search()
    
    return results

if __name__ == "__main__":
    # Direkter Aufruf möglich
    run_hyperparameter_grid_search()