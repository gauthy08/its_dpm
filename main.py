from database.db_manager import create_tables
from scripts.load_data import load_csv_to_db, load_hue_data_to_db, load_dpm_to_db, read_excel_data, load_finrep_y_reference, load_tablestructurehierarchy, load_hue_its  # Import aus dem Ordner 'scripts'
from scripts.merge_data import merge_data, update_merged_data_with_dpm, match_merge_with_reference, find_correct_membername_for_reference, create_output, create_output_corep

def main():
    print("Wähle eine Aktion:")
    print("1: Tabellen(START Konzepte, ITS Base Data) erstellen")
    print("2: START-Konzepte laden")
    print("3: ITS Base Data laden")
    print("4: DPM Daten laden")
    print("5: Finrep-Reference Daten laden")
    print("6: START Konzepte mit ITS Base Data und DPM mergen")
    print("7: Match Merge-Tabelle mit Reference")
    print("8: Finrep - Reference (y-axis) hochladen")
    print("9: DPM_TableStructure hochladen/Hierachie")
    print("10: Create Output")
    print("11: Create Corep-output")
    

    choice = input("Deine Auswahl: ")

    if choice == "1":
        create_tables()
        print("Tabellen wurden erstellt.")
    elif choice == "2":
        load_csv_to_db("data/ISIS-Erhebungsstammdaten1.xlsx")
        print("ISIS-Erhebungsstammdaten wurden in die Datenbank geladen.")
    elif choice == "3":
        #load_hue_data_to_db()#alt
        load_hue_its()#neu
    elif choice == "4":
        load_dpm_to_db("data/qDPM_DataPointCategorisations.csv")
    elif choice == "5":
        #file_path = "data/Annex 3 (FINREP).xlsx"
        file_path = "finrep_references/finrep_reference_x_axis.xlsx"
        read_excel_data(file_path)
        file_path = "finrep_references/finrep_reference_y_axis.xlsx"
        read_excel_data(file_path)
    elif choice == "6":
        merge_data()
        update_merged_data_with_dpm()
    elif choice == "7":
        #match_merge_with_reference()
        find_correct_membername_for_reference()
    elif choice == "8":
        #load_finrep_y_reference("finrep_references/finrep_reference_y_axis.xlsx")
        load_finrep_y_reference("finrep_references/annex_v_result.csv")
    elif choice == "9":
        load_tablestructurehierarchy("data/qDPM_TableStructure.xlsx")
    elif choice == "10":
        create_output()
    elif choice == "11":
        create_output_corep()
    
    else:
        print("Ungültige Auswahl.")

if __name__ == "__main__":
    main()
