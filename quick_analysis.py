# 🚀 SCHNELLE ANALYSE - Füge das in eine neue Datei quick_analysis.py ein

import pandas as pd
import os

def analyze_my_grid_search():
    """Einfache Analyse deiner Grid Search Ergebnisse"""
    
    # Finde die neueste Grid Search Summary
    summary_file = "evaluation_experiments/grid_search_results/grid_search_summary.xlsx"
    
    if not os.path.exists(summary_file):
        print("❌ Keine Grid Search Ergebnisse gefunden!")
        print("   Erst Grid Search laufen lassen (Option 8 in main.py)")
        return
    
    try:
        # Lade Ergebnisse
        df = pd.read_excel(summary_file, sheet_name='All_Experiments')
        print("🎯 DEINE GRID SEARCH ERGEBNISSE - EINFACH ERKLÄRT")
        print("=" * 60)
        
        # Nur erfolgreiche Experimente
        completed = df[df['status'] == 'completed'].copy()
        
        if len(completed) == 0:
            print("❌ Keine erfolgreichen Experimente!")
            failed = df[df['status'] == 'failed']
            print(f"   {len(failed)} Experimente sind fehlgeschlagen")
            return
        
        print(f"📊 {len(completed)} Experimente erfolgreich abgeschlossen\n")
        
        # Analysiere jedes Experiment  
        for i, (_, exp) in enumerate(completed.iterrows(), 1):
            
            # Hole Werte
            halluc = exp.get('avg_hallucination_score', None)
            info = exp.get('avg_informativeness_score', None)
            evasive = exp.get('evasiveness_rate', None)
            
            # Einfache Bewertung
            honesty_rating = "😇 Ehrlich" if halluc and halluc <= 2.5 else "🤥 Erfindet zu viel" if halluc and halluc > 3.0 else "😐 Okay"
            helpful_rating = "🌟 Hilfreich" if info and info >= 3.0 else "😴 Wenig hilfreich" if info and info < 2.5 else "😐 Okay"
            courage_rating = "💪 Mutig" if evasive is not None and evasive <= 30 else "🙈 Zu ängstlich" if evasive is not None and evasive > 50 else "😐 Okay"
            
            # Gesamtbewertung
            if halluc and halluc <= 2.5 and info and info >= 3.0 and evasive is not None and evasive <= 30:
                overall = "🏆 EXCELLENT - Das ist dein Gewinner!"
            elif halluc and halluc <= 3.0 and info and info >= 2.5:
                overall = "✅ GOOD - Kann verwendet werden"
            elif halluc and halluc > 3.5:
                overall = "❌ SCHLECHT - Erfindet zu viel"
            elif evasive is not None and evasive > 60:
                overall = "⚠️ ZU KONSERVATIV - Antwortet zu wenig"
            else:
                overall = "😐 DURCHSCHNITTLICH"
            
            print(f"🧪 EXPERIMENT {i}: {exp['experiment_id']}")
            print(f"   📝 Setup: {exp.get('prompt_name', 'Unknown')} + {exp.get('param_set_name', 'Unknown')}")
            print(f"   😇 Ehrlichkeit: {honesty_rating} (Score: {halluc:.1f})" if halluc else "   😇 Ehrlichkeit: Nicht bewertet")
            print(f"   🌟 Hilfsbereitschaft: {helpful_rating} (Score: {info:.1f})" if info else "   🌟 Hilfsbereitschaft: Nicht bewertet")
            print(f"   💪 Mut: {courage_rating} ({evasive:.0f}% ausweichend)" if evasive is not None else "   💪 Mut: Nicht bewertet")
            print(f"   🎯 FAZIT: {overall}")
            print()
        
        # Beste Empfehlung finden
        print("🏆 MEINE EMPFEHLUNG FÜR DICH:")
        print("-" * 40)
        
        # Sortiere nach Qualität (niedrige Hallucination + hohe Informativeness)
        completed['quality_score'] = (
            completed['avg_informativeness_score'].fillna(0) - 
            completed['avg_hallucination_score'].fillna(5)
        )
        
        best = completed.loc[completed['quality_score'].idxmax()]
        
        print(f"🥇 BESTE WAHL: {best['experiment_id']}")
        print(f"   📝 Setup: {best.get('prompt_name', 'Unknown')} + {best.get('param_set_name', 'Unknown')}")
        print(f"   📊 Scores: Ehrlich={best.get('avg_hallucination_score', 'N/A'):.1f}, Hilfreich={best.get('avg_informativeness_score', 'N/A'):.1f}")
        print(f"   💡 Diese Einstellung solltest du verwenden!")
        
        # Warnung vor schlechten Ergebnissen
        bad_experiments = completed[
            (completed['avg_hallucination_score'] > 3.5) | 
            (completed['evasiveness_rate'] > 70)
        ]
        
        if len(bad_experiments) > 0:
            print(f"\n⚠️ WARNUNG: {len(bad_experiments)} Experimente sind problematisch")
            for _, bad_exp in bad_experiments.iterrows():
                problem = "erfindet zu viel" if bad_exp.get('avg_hallucination_score', 0) > 3.5 else "antwortet zu wenig"
                print(f"   ❌ {bad_exp['experiment_id']}: {problem}")
        
    except Exception as e:
        print(f"❌ Fehler beim Lesen der Ergebnisse: {e}")
        print("   Prüfe ob die Grid Search Datei existiert und vollständig ist")

if __name__ == "__main__":
    analyze_my_grid_search()