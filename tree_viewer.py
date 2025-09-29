import pickle
import os
import json

class ConsoleTreeViewer:
    def __init__(self, pickle_path):
        self.pickle_path = pickle_path
        self.all_trees = self.load_trees()
        self.current_tree = None
        self.current_key = None
    
    def load_trees(self):
        """Lädt das Pickle-File"""
        try:
            with open(self.pickle_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"❌ Datei nicht gefunden: {self.pickle_path}")
            return {}
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            return {}
    
    def clear_screen(self):
        """Löscht den Bildschirm"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def show_menu(self):
        """Zeigt das Hauptmenü"""
        self.clear_screen()
        print("🌳 COREP BAUM VIEWER")
        print("=" * 50)
        print()
        
        if not self.all_trees:
            print("❌ Keine Daten verfügbar!")
            return
        
        print("VERFÜGBARE BÄUME:")
        print("-" * 30)
        
        trees_list = list(self.all_trees.items())
        for i, ((table_code, comp_type), roots) in enumerate(trees_list):
            print(f"{i+1:2d}. {table_code} | {comp_type} ({len(roots)} Wurzeln)")
        
        print()
        print("OPTIONEN:")
        print("  [1-{}] Baum auswählen".format(len(trees_list)))
        print("  [s] Statistiken anzeigen")
        print("  [e] Als JSON exportieren")
        print("  [q] Beenden")
        print()
    
    def show_tree_menu(self):
        """Zeigt das Baum-Menü"""
        if not self.current_tree:
            return
        
        self.clear_screen()
        table_code, comp_type = self.current_key
        print(f"🌳 BAUM: {table_code} - {comp_type}")
        print("=" * 60)
        print()
        
        print("OPTIONEN:")
        print("  [1] Komplette Baumstruktur anzeigen")
        print("  [2] Nur Wurzelknoten anzeigen")
        print("  [3] Statistiken dieses Baums")
        print("  [4] Als JSON exportieren")
        print("  [5] Knoten suchen")
        print("  [b] Zurück zum Hauptmenü")
        print("  [q] Beenden")
        print()
    
    def display_tree_structure(self, detailed=True):
        """Zeigt die Baumstruktur an"""
        if not self.current_tree:
            return
        
        self.clear_screen()
        table_code, comp_type = self.current_key
        print(f"🌳 BAUMSTRUKTUR: {table_code} - {comp_type}")
        print("=" * 70)
        print()
        
        for i, root in enumerate(self.current_tree):
            print(f"📁 WURZEL {i+1}: {root.componentcode}")
            print(f"   {root.componentlabel}")
            print()
            
            if detailed:
                self.print_node_tree(root, "   ")
            else:
                children_count = len(root.children)
                if children_count > 0:
                    print(f"   └── {children_count} Kindknoten")
            
            print()
            print("-" * 50)
            print()
        
        input("\n⏎ Drücken Sie Enter zum Fortfahren...")
    
    def print_node_tree(self, node, prefix="", is_last=True):
        """Druckt einen Knoten-Baum rekursiv"""
        # Begrenze die Tiefe für bessere Lesbarkeit
        if len(prefix) > 20:  # Max ~5 Ebenen
            print(f"{prefix}└── ... (weitere Unterknoten)")
            return
        
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            symbol = "└──" if is_last_child else "├──"
            
            # Kürze lange Labels
            label = child.componentlabel
            if len(label) > 40:
                label = label[:37] + "..."
            
            print(f"{prefix}{symbol} {child.componentcode}: {label}")
            
            # Nächste Ebene
            if child.children:
                next_prefix = prefix + ("    " if is_last_child else "│   ")
                self.print_node_tree(child, next_prefix, is_last_child)
    
    def show_statistics(self, single_tree=False):
        """Zeigt Statistiken"""
        self.clear_screen()
        
        if single_tree and self.current_tree:
            # Statistiken für einen Baum
            table_code, comp_type = self.current_key
            print(f"📊 STATISTIKEN: {table_code} - {comp_type}")
            print("=" * 50)
            print()
            
            total_nodes = 0
            total_leaves = 0
            max_depth = 0
            
            for i, root in enumerate(self.current_tree):
                nodes = self.count_nodes(root)
                leaves = self.count_leaf_nodes(root)
                depth = self.get_max_depth(root)
                
                total_nodes += nodes
                total_leaves += leaves
                max_depth = max(max_depth, depth)
                
                print(f"Wurzel {i+1}: {root.componentcode}")
                print(f"  Label: {root.componentlabel}")
                print(f"  Knoten gesamt: {nodes}")
                print(f"  Blattknoten: {leaves}")
                print(f"  Maximale Tiefe: {depth}")
                print()
            
            print("GESAMTSTATISTIK:")
            print(f"  Wurzelknoten: {len(self.current_tree)}")
            print(f"  Knoten gesamt: {total_nodes}")
            print(f"  Blattknoten: {total_leaves}")
            print(f"  Maximale Tiefe: {max_depth}")
            
        else:
            # Statistiken für alle Bäume
            print("📊 GESAMTSTATISTIKEN")
            print("=" * 50)
            print()
            
            for (table_code, comp_type), roots in self.all_trees.items():
                total_nodes = sum(self.count_nodes(root) for root in roots)
                print(f"📈 {table_code} - {comp_type}")
                print(f"   Wurzeln: {len(roots)}, Knoten: {total_nodes}")
        
        print()
        input("⏎ Drücken Sie Enter zum Fortfahren...")
    
    def search_nodes(self):
        """Sucht nach Knoten"""
        if not self.current_tree:
            return
        
        self.clear_screen()
        table_code, comp_type = self.current_key
        print(f"🔍 KNOTENSUCHE: {table_code} - {comp_type}")
        print("=" * 50)
        print()
        
        search_term = input("Suchbegriff eingeben (Code oder Label): ").strip().lower()
        if not search_term:
            return
        
        print(f"\nSuche nach: '{search_term}'")
        print("-" * 30)
        
        found_nodes = []
        for root in self.current_tree:
            self.find_nodes_recursive(root, search_term, found_nodes, "")
        
        if found_nodes:
            print(f"\n✅ {len(found_nodes)} Treffer gefunden:")
            print()
            for path, node in found_nodes:
                print(f"📍 {path}")
                print(f"   Code: {node.componentcode}")
                print(f"   Label: {node.componentlabel}")
                print(f"   Level: {node.level}")
                print()
        else:
            print("❌ Keine Treffer gefunden.")
        
        input("\n⏎ Drücken Sie Enter zum Fortfahren...")
    
    def find_nodes_recursive(self, node, search_term, results, path):
        """Rekursive Knotensuche"""
        current_path = f"{path} > {node.componentcode}" if path else node.componentcode
        
        # Prüfe ob dieser Knoten dem Suchbegriff entspricht
        if (search_term in node.componentcode.lower() or 
            search_term in node.componentlabel.lower()):
            results.append((current_path, node))
        
        # Suche in Kindern
        for child in node.children:
            self.find_nodes_recursive(child, search_term, results, current_path)
    
    def export_json(self, single_tree=False):
        """Exportiert als JSON"""
        if single_tree and self.current_tree:
            table_code, comp_type = self.current_key
            filename = f"{table_code}_{comp_type.replace(' ', '_')}.json"
            
            data = {
                "table_code": table_code,
                "component_type": comp_type,
                "roots": [self.node_to_dict(root) for root in self.current_tree]
            }
        else:
            filename = "all_trees.json"
            data = {}
            for (table_code, comp_type), roots in self.all_trees.items():
                key = f"{table_code}_{comp_type.replace(' ', '_')}"
                data[key] = [self.node_to_dict(root) for root in roots]
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Exportiert nach: {filename}")
        except Exception as e:
            print(f"❌ Export-Fehler: {e}")
        
        input("\n⏎ Drücken Sie Enter zum Fortfahren...")
    
    def node_to_dict(self, node):
        """Konvertiert Node zu Dictionary"""
        return {
            "componentcode": node.componentcode,
            "componentlabel": node.componentlabel,
            "level": node.level,
            "order": node.order,
            "headerflag": node.headerflag,
            "ordinateid": node.ordinateid,
            "children": [self.node_to_dict(child) for child in node.children]
        }
    
    def count_nodes(self, node):
        """Zählt Knoten rekursiv"""
        count = 1
        for child in node.children:
            count += self.count_nodes(child)
        return count
    
    def count_leaf_nodes(self, node):
        """Zählt Blattknoten"""
        if not node.children:
            return 1
        return sum(self.count_leaf_nodes(child) for child in node.children)
    
    def get_max_depth(self, node, current_depth=0):
        """Ermittelt maximale Tiefe"""
        if not node.children:
            return current_depth
        return max(self.get_max_depth(child, current_depth + 1) for child in node.children)
    
    def run(self):
        """Hauptschleife"""
        if not self.all_trees:
            print("❌ Keine Daten verfügbar!")
            return
        
        trees_list = list(self.all_trees.items())
        
        while True:
            if self.current_tree is None:
                # Hauptmenü
                self.show_menu()
                choice = input("Ihre Wahl: ").strip().lower()
                
                if choice == 'q':
                    print("👋 Auf Wiedersehen!")
                    break
                elif choice == 's':
                    self.show_statistics()
                elif choice == 'e':
                    self.export_json()
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(trees_list):
                        self.current_key, self.current_tree = trees_list[idx]
                    else:
                        print("❌ Ungültige Auswahl!")
                        input("⏎ Drücken Sie Enter zum Fortfahren...")
            
            else:
                # Baum-Menü
                self.show_tree_menu()
                choice = input("Ihre Wahl: ").strip().lower()
                
                if choice == 'q':
                    print("👋 Auf Wiedersehen!")
                    break
                elif choice == 'b':
                    self.current_tree = None
                    self.current_key = None
                elif choice == '1':
                    self.display_tree_structure(detailed=True)
                elif choice == '2':
                    self.display_tree_structure(detailed=False)
                elif choice == '3':
                    self.show_statistics(single_tree=True)
                elif choice == '4':
                    self.export_json(single_tree=True)
                elif choice == '5':
                    self.search_nodes()
                else:
                    print("❌ Ungültige Auswahl!")
                    input("⏎ Drücken Sie Enter zum Fortfahren...")

def main():
    pickle_path = os.path.join("tree_structures", "baumstruktur_COREP_3_2.pkl")
    
    print("🌳 COREP Konsolen-Viewer startet...")
    print()
    
    viewer = ConsoleTreeViewer(pickle_path)
    viewer.run()

if __name__ == "__main__":
    main()