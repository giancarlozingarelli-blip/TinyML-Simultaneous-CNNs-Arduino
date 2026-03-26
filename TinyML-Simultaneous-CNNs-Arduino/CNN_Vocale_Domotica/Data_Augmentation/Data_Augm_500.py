import os # Mi serve per esplorare le cartelle e i percorsi dei file nel sistema operativo
import shutil # Strumento fondamentale per cancellare o copiare brutalmente intere cartelle o file
import random # Lo uso per estrarre a sorte i file da copiare o per decidere quale "mutazione" applicare all'audio
from pydub import AudioSegment # La libreria magica per manipolare fisicamente il file audio (volume, silenzi, ritagli)

# ==========================================
# IMPOSTAZIONI GLOBALI
# ==========================================
# Definisco le cartelle: prendo i file già puliti e tagliati a 1 secondo, e genero il dataset aumentato
INPUT_DIR = "dataset_audio_Dom_processed"
OUTPUT_DIR = "dataset_audio_Dom_augmented"

# Il target assoluto: decido che tutte le classi dovranno avere questo numero esatto di campioni.
# Se ne hanno di più elimino l'eccesso (Undersampling), se ne hanno di meno li invento (Oversampling). Questo uccide il bias della rete.
UNIVERSAL_TARGET = 500

# ==========================================
# BLOCCO DATA AUGMENTATION
# ==========================================
# Prende un singolo file audio perfetto e lo "sporca" o lo sposta leggermente nel tempo. 
# Serve per insegnare alla rete che la voce può essere più alta, più bassa, o iniziare con un leggero ritardo.
def augment_audio(audio):
    # Estraggo un numero a caso da 0 a 3 (quindi 4 opzioni possibili) per decidere quale mutazione applicare
    choice = random.randint(0, 3)   
    
    if choice == 0:
        # Mutazione 0: Alzo il volume a caso tra +2.0 e +5.0 decibel
        return audio + random.uniform(2.0, 5.0)
        
    elif choice == 1:
        # Mutazione 1: Abbasso il volume a caso tra -2.0 e -5.0 decibel
        return audio - random.uniform(2.0, 5.0)
        
    elif choice == 2:
        # Mutazione 2: Time-shifting (spostamento in avanti). Fondamentale per parole ostiche come "Guida".
        shift_ms = random.randint(50, 150) # Decido di quanto ritardare la voce (da 50 a 150 millisecondi)
        silence = AudioSegment.silent(duration=shift_ms) # Creo un pezzo di silenzio puro lungo quanto il ritardo
        shifted = silence + audio # Attacco il silenzio all'inizio, spingendo la voce in avanti
        return shifted[:len(audio)] # Il file ora è più lungo di 1 sec, quindi taglio la coda per farlo tornare alla lunghezza originale
        
    else:
        # Mutazione 3: Time-shifting (spostamento all'indietro). La parola inizia in anticipo.
        shift_ms = random.randint(50, 150) # Decido l'anticipo
        silence = AudioSegment.silent(duration=shift_ms) # Creo il blocco di silenzio
        shifted = audio[shift_ms:] + silence # Taglio via la testa dell'audio vero e appiccico il silenzio alla fine per tappare il buco
        return shifted # Ritorno l'audio che ora è di nuovo lungo esattamente 1 secondo

# ==========================================
# BLOCCO MOTORE PRINCIPALE DI BILANCIAMENTO
# ==========================================
# Scansiona le cartelle. Se i file sono troppi li taglia, se sono pochi avvia il ciclo di mutazione per crearne di nuovi.
def process_augmentation():
    # AZZERAMENTO: Elimino la vecchia cartella di output per non mescolare training vecchi e nuovi
    if os.path.exists(OUTPUT_DIR):
        print(f" Trovata cartella precedente '{OUTPUT_DIR}'. Eliminazione in corso...")
        shutil.rmtree(OUTPUT_DIR) # Cancella l'intera cartella e tutto quello che c'è dentro

    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ricrea la cartella di output principale, ora vuota
    print(f" Creata nuova cartella vuota: {OUTPUT_DIR}\n")

    # Scorro le cartelle delle classi (clima, luce, rumore...) una per una
    for label in os.listdir(INPUT_DIR):
        input_label_dir = os.path.join(INPUT_DIR, label) # Costruisco il percorso esatto di questa cartella
        
        # Se c'è un file estraneo (tipo .DS_Store o simili) invece di una cartella, lo salto
        if not os.path.isdir(input_label_dir):
            continue
            
        # Creo la sottocartella speculare nella cartella di destinazione
        output_label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(output_label_dir, exist_ok=True)
            
        # Genero una lista contenente tutti i nomi dei file .wav originali presenti in questa classe
        original_files = [f for f in os.listdir(input_label_dir) if f.endswith(".wav")]
        num_originals = len(original_files) # Conto quanti file ci sono
        
        # Se per qualche motivo la cartella è vuota, avviso l'utente e passo alla classe successiva
        if num_originals == 0:
            print(f"Attenzione: Nessun file in {label}")
            continue
            
        print(f"\nElaborazione classe: '{label}' (File originali: {num_originals})")
        
        # ---------------------------------------------------------------------
        # CASO 1: UNDERSAMPLING (Classe con troppi file, es: se background_noises ha più di 500 campioni,ma non è questo il caso in quanto ne ho inseriti circa 200 massimo bilancaimento del dataset di partenza)
        # Può essere omesso come ciclo if , ma meglio metterlo per sicurezza e scalabilità del progetto
        # ---------------------------------------------------------------------
        if num_originals > UNIVERSAL_TARGET:
            print(f"--> Riduzione da {num_originals} a {UNIVERSAL_TARGET} campioni (Undersampling)...")
            
            # Peschiamo dal mazzo esattamente il numero di file che ci serve (500), scartando gli altri
            selected_files = random.sample(original_files, UNIVERSAL_TARGET)
            
            for filename in selected_files:
                # Copio brutalmente il file da una cartella all'altra. 
                # Uso shutil.copy2 perché fa una copia del file a livello di sistema operativo ed è istantaneo (non apro l'audio in RAM)
                shutil.copy2(os.path.join(input_label_dir, filename), os.path.join(output_label_dir, filename))
                
        # ---------------------------------------------------------------------
        # CASO 2: OVERSAMPLING E AUGMENTATION (Classe con pochi file, es: guida, ma in realtà tutte le classi nel mio caso )
        # ---------------------------------------------------------------------
        else:
            # Primo step: mi assicuro di salvare tutti i file originali buoni, copiandoli nella nuova cartella
            for filename in original_files:
                shutil.copy2(os.path.join(input_label_dir, filename), os.path.join(output_label_dir, filename))
                
            # Calcolo quanti file sintetici mi mancano per arrivare all'obiettivo  (es: 500 - 150 = 350 file da inventare)
            num_to_generate = UNIVERSAL_TARGET - num_originals
            
            if num_to_generate > 0:
                print(f"--> Generazione di {num_to_generate} file sintetici per raggiungere quota {UNIVERSAL_TARGET}...")
                
                # Faccio un ciclo che si ripete esattamente per il numero di file mancanti
                for i in range(num_to_generate):
                    # Pesco a caso uno dei miei file originali per usarlo come base della mutazione
                    base_file = random.choice(original_files)
                    
                    # Lo carico in RAM con Pydub in modo da poterlo alterare
                    base_audio = AudioSegment.from_file(os.path.join(input_label_dir, base_file))
                    
                    # Passo l'audio alla mia funzione di Data Augmentation che lo altera e me lo restituisce "sporcato"
                    augmented_audio = augment_audio(base_audio)
                    
                    # Mi invento un nome nuovo per il file clonato, così non sovrascrivo l'originale
                    new_filename = f"aug_{i}_{base_file}"
                    
                    # Esporto fisicamente il nuovo file mutato nella cartella finale
                    augmented_audio.export(os.path.join(output_label_dir, new_filename), format="wav")
            else:
                # Questo scatta se per puro miracolo la cartella aveva già esattamente 500 file 
                print(f"--> Target di {UNIVERSAL_TARGET} perfettamente raggiunto.")
            
    # Messaggio finale di vittoria quando il ciclo for su tutte le cartelle è concluso
    print("\n Bilanciamento e Data Augmentation (con 8 classi) completati con successo!")

# Controllo standard di Python: esegue la funzione principale solo se sto lanciando questo script direttamente
if __name__ == "__main__":
    process_augmentation()