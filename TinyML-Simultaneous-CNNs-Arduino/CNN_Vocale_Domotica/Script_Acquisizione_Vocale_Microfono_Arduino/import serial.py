import serial # Importo la libreria PySerial per far comunicare Python e Arduino tramite cavo USB
import wave # La libreria fondamentale per "impacchettare" i dati grezzi e trasformarli in un vero file .wav ascoltabile
import os # Mi serve per esplorare il disco rigido e creare le cartelle
import time # Lo uso per mettere in pausa lo script (ci servirà all'avvio)

# ==========================================
# BLOCCO 1: IMPOSTAZIONI DEL PROGETTO
# ==========================================
# Qui metto le due uniche variabili che dovrò cambiare manualmente ogni volta 
# che voglio registrare una nuova parola.

# Inserisci la porta esatta del tuo Arduino (es. COM7, COM12)
SERIAL_PORT = "COM7" # È la "porta fisica" a cui è attaccato il cavo USB, viene mostrata nell' IDE

# Inserisci la parola che stai per registrare
CLASS_NAME = "prova"  # Diventerà sia il nome della cartella che il prefisso dei file
# ==========================================

# Regole di comunicazione 
BAUD_RATE = 115200 # La velocità  della porta USB (deve essere identica al Serial.begin di Arduino)
# 32.000 campioni (2 secondi di audio) moltiplicati per 2 byte (16-bit) = 64.000 byte totali.
TOTAL_BYTES = 64000 

# Crea la cartella di destinazione (se non esiste già)
OUTPUT_DIR = CLASS_NAME # Imposto il nome della cartella uguale al nome della parola
os.makedirs(OUTPUT_DIR, exist_ok=True) # Dico al sistema operativo di creare la cartella. Se esiste già, non fa niente e non va in crash.

# =============================================
# BLOCCO 2: CONNESSIONE HARDWARE
# =============================================
# Apertura del canale di comunicazione con la scheda.

print(f" Connessione all'Arduino sulla porta {SERIAL_PORT}...")
try:
    # Apro fisicamente la porta Seriale. Se l'Arduino non risponde entro 5 secondi, va in timeout
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    
    #  Quando Python apre la porta Seriale, l'Arduino si riavvia fisicamente (è una sua caratteristica hardware).
    # Se gli mando subito un comando, lo perdo perché lui si sta accendendo. Lo faccio aspettare 2 secondi.
    time.sleep(2)  
    print(" Connesso!")
except Exception as e:
    # Se Python non riesce ad aprire la porta (spesso perché va chiuso il Monitor Seriale nell'IDE di Arduino, lasciato aperto o aperto per sbaglio)
    print(f" Errore di connessione. Controlla la porta COM e chiudi il Monitor Seriale di Arduino.\nErrore: {e}")
    exit() # Spengo lo script 

# =====================================================================
# BLOCCO 3: PREPARAZIONE AL SALVATAGGIO
# =====================================================================
# Calcolo a che numero di file siamo arrivati per non sovrascrivere le vecchie registrazioni,così se riavvio lo script in
#un secondo momento posso continuare dal file successivo all' ultimo che ho slavato.

# Esploro la cartella e creo una lista di tutti i file che iniziano con la mia parola e finiscono in .wav
esistenti = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(CLASS_NAME) and f.endswith(".wav")]
# Conto quanti file ci sono e aggiungo 1. Così, se ho 40 file, il prossimo si chiamerà "prova_041.wav"
count = len(esistenti) + 1

print(f"\n PRONTO A REGISTRARE PER LA CLASSE: '{CLASS_NAME}'")
print(f" I file verranno salvati nella cartella: ./{OUTPUT_DIR}/")
print("======================================================")


# =====================================================================
# BLOCCO 4: LOOP DI REGISTRAZIONE 
# =====================================================================
# Qui Python dà il via all'Arduino, raccoglie la registrazione e la salva.

try:
    while True: # Entro in un loop infinito. Uscirò solo quando premerò CTRL+C sulla tastiera.
        
        # Metto in pausa lo script e aspetto che l'utente prema il tasto INVIO sulla tastiera per partire. 
        # Uso {count:03d} per forzare 3 zeri (es: 001, 002, 015) in modo che i file siano ordinati bene alfabeticamente.
        # Se ho intenzioni di registrare più di 999 campioni, devo forzare 4 zeri
        input(f"Premi INVIO per registrare '{CLASS_NAME}_{count:03d}.wav' (o CTRL+C per uscire)... ")
        
        # Mando la lettera "R" (codificata in byte con la 'b' davanti) lungo il cavo USB per svegliare l'Arduino
        ser.write(b'R')
        
        raw_data = b'' # Preparo un contenitore di byte vuoto dove accumulerò il suono
        # end="\r" serve a far sì che il testo resti sulla stessa riga senza andare a capo
        print("Registrazione in corso...", end="\r") 
        
        # Inizio a prednere dati dal cavo USB finché non arrivo esattamente al peso che mi aspetto, 64000 byte
        while len(raw_data) < TOTAL_BYTES:
            # Leggo dalla USB solo i byte che mi mancano per arrivare a quota 64000
            blocco = ser.read(TOTAL_BYTES - len(raw_data))
            
            # Se la lettura restituisce il vuoto, significa che il cavo si è staccato o l'Arduino si è bloccato
            if not blocco:
                print("\n L'Arduino non risponde.")
                break # Rompo il ciclo while per non restare bloccato qui in eterno
                
            # Aggiungo il blocco appena scaricato al mio contenitore principale
            raw_data += blocco
            
        # Quando il contenitore è esattamente pieno (64.000 byte)...
        if len(raw_data) == TOTAL_BYTES:
            
            # Unisco il nome della cartella col nome del file (es: "prova/prova_001.wav")
            filename = os.path.join(OUTPUT_DIR, f"{CLASS_NAME}_{count:03d}.wav")
            
            # Apro un file fisico sul computer in modalità "scrittura binaria" ('wb') usando la libreria wave
            with wave.open(filename, 'wb') as wf:
                
                # INIZIO A INSERIRE LE METADATI (L'intestazione del file WAV)
                wf.setnchannels(1)           # Dico a Windows/Mac che è un audio Mono (1 canale)
                wf.setsampwidth(2)           # Dico che ogni campione pesa 2 byte (cioè audio a 16-bit)
                wf.setframerate(16000)       # Dico che la velocità di riproduzione è 16.000 Hertz
                
                # Riverso tutti i dati grezzi dentro il file
                wf.writeframes(raw_data)
                
            # Il file così si slava e  printo il successo sovrascrivendo la vecchia riga
            print(f" Salvato: {filename}   ")
            count += 1 # Aggiorno il contatore per il file successivo (es: passo da 001 a 002)

# =====================================================================
# BLOCCO 5: CHIUSURA
# =====================================================================
except KeyboardInterrupt:
    # Se l'utente preme CTRL+C nel terminale, catturo l'evento per non far comparire errori rossi a schermo
    print("\n\n Registrazione terminata dall'utente.")
finally:
    # Che il programma sia finito o sia crashato, questa riga viene eseguita in ogni caso
    ser.close() # Chiudo la porta USB. Se la lascio aperta, non posso utilizzarla per caricare altri sketch Ide sul microcontrollore