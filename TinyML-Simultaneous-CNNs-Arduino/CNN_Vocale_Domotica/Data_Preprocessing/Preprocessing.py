import os # Importa il modulo per interagire con il sistema operativo (creare cartelle, gestire i percorsi dei file)
import shutil # Importa il modulo per operazioni avanzate sui file (ci serve per cancellare intere cartelle in un colpo solo)
import random # Importa il modulo per generare numeri casuali (lo useremo per pescare frammenti di rumore a caso)
from pydub import AudioSegment # Importa la libreria principale per manipolare i file audio (tagliare, unire, convertire)

# ==========================================
# 1. PARAMETRI GLOBALI DEL PROGETTO
# ==========================================
# Setup fisso per Arduino: se non gli do un WAV a 16kHz, Mono e di esattamente 1 secondo, il modello TFLite crasha.

TARGET_DURATION_MS = 1000  # Fissa la durata esatta che vogliamo ottenere: 1000 millisecondi (cioè 1 secondo)
TARGET_CHANNELS = 1        # Fissa il numero di canali audio a 1, forzando il formato Mono (anziché Stereo), perchè l' arduino ha 1 solo microfono
TARGET_SAMPLE_RATE = 16000 # Fissa la frequenza di campionamento a 16000 Hz (lo standard hardware richiesto dal PDM dell'Arduino)

INPUT_DIR = "dataset_audio_Dom"              # Definisce il nome della cartella dove teniamo i file audio originali (grezzi)
OUTPUT_DIR = "dataset_audio_Dom_processed"   # Definisce il nome della cartella dove lo script salverà i file puliti e pronti

# ==========================================
# 2. METODI DI ESTRAZIONE E PADDING
# ==========================================

def extract_unified_window(audio):
    # Trova la parola dentro l'audio. Se è corto aggiungo silenzio alla fine, se è lungo cerco il picco di volume (RMS) per isolare la voce.
    
    # Controlla se la lunghezza dell'audio in input è strettamente minore di 1 secondo (1000 ms)
    if len(audio) < TARGET_DURATION_MS:
        # Calcola esattamente quanti millisecondi mancano per arrivare a 1 secondo tondo
        missing_ms = TARGET_DURATION_MS - len(audio)
        # Genera un blocco di silenzio puro lungo "missing_ms" e lo attacca alla fine dell'audio originale
        return audio + AudioSegment.silent(duration=missing_ms)
        
    # Controlla se l'audio dura già 1 secondo esatto (caso del dataset scaricato da me ad esempio, invece i dati che ho registrato durano tutti 2 secondi)
    elif len(audio) == TARGET_DURATION_MS:
        # Ritorna l'audio così com'è, non c'è bisogno di toccarlo
        return audio
        
    # Se l'audio dura più di 1 secondo (il caso più probabile quando si registra dal vivo)
    else:
        max_energy = -1 # Inizializza la variabile che terrà traccia del volume massimo trovato (RMS) partendo da un valore bassissimo
        best_start = 0  # Inizializza la variabile che salverà l'istante di tempo (in ms) in cui inizia la parola
        step_ms = 10    # Definisce il "passo": la nostra finestra di analisi scivolerà in avanti di 10 millisecondi alla volta
        
        # Scorro l'audio a pezzetti di 10ms e mi tengo il blocco di 1 secondo dove si sente più forte
        # Il ciclo "for" parte da 0 e si ferma prima dell'ultimo secondo disponibile per evitare di andare "fuori bordo"
        for i in range(0, len(audio) - TARGET_DURATION_MS + 1, step_ms):
            # Ritaglia provvisoriamente una fetta (chunk) di audio lunga esattamente 1 secondo, partendo dal millisecondo 'i'
            chunk = audio[i:i + TARGET_DURATION_MS]
            # Controlla se l'energia acustica (volume RMS) di questa fetta provvisoria batte il record precedente
            if chunk.rms > max_energy:
                max_energy = chunk.rms # Aggiorna il record di energia
                best_start = i         # Salva 'i' come nuovo istante di inizio perfetto per il taglio
                
        # Alla fine del ciclo, ritaglia e restituisce il secondo esatto che contiene l'energia (la voce) più alta
        return audio[best_start:best_start + TARGET_DURATION_MS]


def extract_random_section(audio):
    # Usato solo per i rumori di fondo. Pesco un secondo a caso dal file lungo per dare un po' di varietà alla rete.
    
    # Se il file di rumore è più corto o uguale a 1 secondo, lo "riempio" di silenzio fino ad arrivare a 1000ms
    if len(audio) <= TARGET_DURATION_MS:
        missing_ms = TARGET_DURATION_MS - len(audio) # Calcola il tempo mancante
        return audio + AudioSegment.silent(duration=missing_ms) # Ritorna il file allungato col silenzio
    else:
        # Se il file è lungo, calcola il punto massimo in cui posso iniziare a tagliare senza che il taglio finisca oltre la fine del file
        max_start = len(audio) - TARGET_DURATION_MS
        # Sceglie un numero a caso tra 0 e il punto massimo calcolato prima
        random_start = random.randint(0, max_start)
        # Ritaglia 1 secondo di audio partendo da quel punto estratto a sorte
        return audio[random_start:random_start + TARGET_DURATION_MS]


# ==========================================
# 3. PIPELINE PRINCIPALE DI ELABORAZIONE
# ==========================================

def process_audio_files():
    # Controlla se la cartella di destinazione (quella dei file processati) esiste già
    if os.path.exists(OUTPUT_DIR):
        print(f" Rilevata cartella esistente, la pulisco: {OUTPUT_DIR}") # Avvisa l'utente nel terminale
        shutil.rmtree(OUTPUT_DIR) # Elimino l'intera cartella per evitare di mischiare file vecchi e nuovi
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Crea (o ricrea) la cartella di output pulita
    
    # Scansiona tutto ciò che c'è dentro la cartella di input (dataset grezzo) e prende un nome alla volta (es: "clima")
    for label in os.listdir(INPUT_DIR):
        # Unisce il nome della cartella principale con il nome della classe per ottenere il percorso completo (es: "dataset/clima")
        label_dir = os.path.join(INPUT_DIR, label)
        # Se per caso l'elemento analizzato è un file orfano e non una cartella, lo salta con "continue"
        if not os.path.isdir(label_dir):
            continue
            
        # Crea il percorso speculare dove salvare i file elaborati (es: "dataset_processed/clima")
        output_label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(output_label_dir, exist_ok=True) # Crea fisicamente questa cartella nel sistema
        
        # Decide quale dei due algoritmi matematici usare in base al nome della cartella
        # Potrei anche non creare la variabile method ed usare dopo direttamente 
        #' if label == "background_noises" ' e 'else:' , ma cosi rendo il codice piu modificabile se 
        # voglio aggiungere cartelle altre cartelle/classi in futuro con altri tipi
        # di rumori di sottofonodo, ad esempio se voglio scalare aggiungendo una cartella/classe
        # con i soli silenzi del microfono del mio arduino
        
        if label == "background_noises":
            method = "RANDOM" # Se è la cartella dei rumori, usa il metodo casuale
        else:
            method = "UNIFIED_RMS" # Per tutte le altre parole, usa il rilevamento del volume
            
        print(f"Processando [{label}] con metodo: {method}") # Stampa l'inizio del lavoro per questa classe
        
        # Inizia a navigare in profondità dentro la cartella della classe, trovando tutti i file e le eventuali sottocartelle
        for root, dirs, files in os.walk(label_dir):
            # Prende uno ad uno tutti i file trovati
            for filename in files:
                # Controlla se il file finisce con un'estensione audio valida (ignorando eventuali file di sistema nascosti)
                if filename.lower().endswith((".wav", ".mp3", ".opus", ".ogg")):
                    # Costruisce il percorso totale del file originale
                    input_file_path = os.path.join(root, filename)
                    # Divide il nome del file dalla sua estensione e si tiene solo il nome (es: "audio1.wav" -> "audio1")
                    name_without_ext = os.path.splitext(filename)[0] 
                    # Calcola il percorso "relativo" per capire se il file era infilato dentro ulteriori sottocartelle
                    rel_path = os.path.relpath(root, label_dir)          
                    
                    # Inventa il nome per il nuovo file. Se era in una sottocartella, mette il nome della sottocartella per evitare doppioni
                    dest_name = f"{name_without_ext}.wav" if rel_path == "." else f"{rel_path.replace(os.sep, '_')}_{name_without_ext}.wav"
                    # Costruisce il percorso completo dove andrà salvato il file finale processato
                    output_file_path = os.path.join(output_label_dir, dest_name)
                    
                    try:
                        # Carico l'audio, forzo i 16kHz Mono e faccio il taglio
                        # Passa il percorso del file originale alla libreria Pydub, che lo decodifica e lo mette in RAM
                        audio = AudioSegment.from_file(input_file_path)
                        
                        # Concatena due comandi Pydub: trasforma in Mono (1 canale) e fa il resample a 16000 Hz
                        audio = audio.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_SAMPLE_RATE)
                        
                        # Controlla quale stringa contiene la variabile "method" per questa classe
                        if method == "RANDOM":
                            # Chiama la funzione di estrazione casuale passando l'audio e salva il risultato
                            audio_processed = extract_random_section(audio)
                        else:
                            # Chiama la funzione di ricerca energetica RMS passando l'audio e salva il risultato
                            audio_processed = extract_unified_window(audio)
                                                        
                        # Esporto. Il formato pcm_s16le è fondamentale, altrimenti il MicroFrontend su Arduino legge spazzatura e non capisce nulla.
                        # Scrive l'audio processato sul disco rigido, forzando l'estensione wav e il codec a 16-bit signed integer little-endian
                        audio_processed.export(output_file_path, format="wav", parameters=["-acodec", "pcm_s16le"])
                        
                    except Exception as e:
                        # Se il file originale è rotto o illeggibile, il "try" fallisce. Catturiamo l'errore, lo stampiamo e proseguiamo col file successivo
                        print(f"   Errore sull'estrazione di {filename}: {e}")
                        
    # Finita tutta l'esplorazione e la conversione, stampa il messaggio finale
    print("\n Elaborazione offline conclusa con successo!")

# Variabile speciale di Python. Controlla se stiamo lanciando lo script direttamente da terminale
if __name__ == "__main__":
    process_audio_files() # Avvia concretamente tutto il motore di elaborazione