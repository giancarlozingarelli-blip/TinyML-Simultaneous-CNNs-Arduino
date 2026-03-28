import os # Mi serve per interagire con il sistema operativo e i percorsi dei file
import shutil # Libreria per eliminare cartelle piene 
from PIL import Image, ImageOps # Il motore principale per caricare, ridimensionare e manipolare le nostre foto

# =====================================================================
# BLOCCO 1: IMPOSTAZIONI GLOBALI E PARAMETRI HARDWARE
# =====================================================================

INPUT_DIR = 'Dataset' # Punta alla cartella grezza dove ho raccolto tutte le mie foto originali

OUTPUT_DIR = 'Dataset_processed' # La cartella pulita che il programma creerà da zero per i dati pronti

# Questi numeri devono coincidere millimetricamente con i parametri C++ della scheda (vincoli SRAM di Arduino)
IMG_WIDTH = 96 # Larghezza finale dell'immagine ritagliata (96 pixel)
IMG_HEIGHT = 96 # Altezza finale dell'immagine ritagliata (96 pixel)


# =====================================================================
# BLOCCO 2: MOTORE DI PREPROCESSING (PILLOW) E RITAGLIO CENTRALE
# =====================================================================

def process_images(): # Funzione "Master" che incatena tutti i passaggi di elaborazione
    
    if os.path.exists(OUTPUT_DIR): # Controllo se esiste già una vecchia cartella di output
        print(f"[*] Rilevata cartella esistente '{OUTPUT_DIR}'. Eliminazione in corso...") # Avviso l'utente
        shutil.rmtree(OUTPUT_DIR) # Distrugoo la vecchia cartella per evitare il "Data Leakage" e mescolamenti
    
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ricreo la cartella di output pulita e vuota
    print(f"[*] Creata nuova cartella di destinazione: '{OUTPUT_DIR}'\n") 

    # Esploro la cartella principale per trovare le sottocartelle delle mie classi (es. ostacoli, via_libera)
    for label in os.listdir(INPUT_DIR): # Scorro ogni elemento presente nella cartella Dataset
        input_label_dir = os.path.join(INPUT_DIR, label) # Unisco il percorso per entrare nella cartella specifica

        if not os.path.isdir(input_label_dir): # Se trovo un file sciolto anziché una cartella...
            continue # ...lo ignoro e passo al prossimo 

        output_label_dir = os.path.join(OUTPUT_DIR, label) # Preparo il percorso della rispettiva cartella di destinazione
        os.makedirs(output_label_dir, exist_ok=True) # Creo la cartella per questa specifica classe anche nell'output

        print(f"Inizio elaborazione profonda per la classe: [{label}]...") # Stampo a che punto sono
        count = 0 # Inizializzo un contatore per rinominare i file in modo ordinato (es. 00000, 00001)

        # Scavo a fondo in tutte le eventuali sottocartelle di questa classe
        for root, _, files in os.walk(input_label_dir): # os.walk esplora in profondità
            for filename in files: # Prendo uno ad uno i file trovati
                
                # Filtro i file per essere sicuro di processare solo immagini con formati validi
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    input_path = os.path.join(root, filename) # Costruisco il percorso completo del file grezzo
                    
                    # Rinomino il file per standardizzarlo ed evitare sovrascritture in caso di file omonimi
                    nuovo_nome_file = f"{label}_{count:05d}.jpg" # Aggiungo zeri per ordinamento alfabetico corretto
                    output_path = os.path.join(output_label_dir, nuovo_nome_file) # Percorso finale di salvataggio

                    try: # Uso un blocco try per evitare che un'immagine corrotta faccia crashare tutto il loop
                        with Image.open(input_path) as img: # Apro l'immagine bloccandola in RAM in modo sicuro
                            
                            img_gray = img.convert('L') # Convertiamo in Scala di Grigi (luminanza a 1 canale)
                            
                            # STEP 1: SIMULAZIONE SENSORE
                            # Ridimensioniamo ai pixel nativi del sensore QCIF (176x144) Per replicare il più possibile le operazioni
                            # fatte dall' Arduino
                            img_qcif = img_gray.resize((176, 144)) # Emulazione hardware ottico della fotocamera
                            
                            # STEP 2: RITAGLIO CENTRALE MATEMATICO
                            width, height = img_qcif.size # Estraggo le dimensioni attuali del tensore
                            left = (width - IMG_WIDTH) / 2 # Calcolo l'origine asse X per il bounding box
                            top = (height - IMG_HEIGHT) / 2 # Calcolo l'origine asse Y per il bounding box
                            right = (width + IMG_WIDTH) / 2 # Calcolo il limite di estensione asse X
                            bottom = (height + IMG_HEIGHT) / 2 # Calcolo il limite di estensione asse Y
                            
                            # Eseguo il ritaglio centrale usando le coordinate calcolate
                            img_cropped = img_qcif.crop((left, top, right, bottom)) 
                            
                            # Salviamo l'immagine processata nella nuova cartella in formato JPG
                            img_cropped.save(output_path) 
                            count += 1 # Incremento l'indice per il prossimo file
                            
                    except Exception as e: # Costrutto di debugging: se il file è illeggibile...
                        print(f" -> Errore nell'elaborazione del file {input_path}: {e}") # ...segnalo l'errore senza fermare lo script
        
        print(f"    Completata! {count} immagini salvate in '{output_label_dir}'.") # Summary metrico di fine classe

    # =====================================================================
    # BLOCCO 3: FINE ESECUZIONE
    # =====================================================================
    print("\n=======================================================")
    print(" PREPROCESSING OFFLINE COMPLETATO CON SUCCESSO!") # Flag terminale di successo globale
    print("=======================================================")

if __name__ == "__main__": # Controllo modulo per capire se lo script è eseguito direttamente
    process_images() # Invocazione esplicita della routine principale