import os # Mi serve per gestire i percorsi di sistema
import pathlib # Libreria comodissima per esplorare le cartelle del dataset
import numpy as np # Gestisce gli array e la matematica
import tensorflow as tf # Il motore principale di intelligenza artificiale
from tensorflow import keras # L'interfaccia ad alto livello di TensorFlow per creare la rete neurale
from tensorflow.keras import layers, models # I layer per costruire l'architettura della rete
import matplotlib.pyplot as plt # La libreria per disegnare i grafici di addestramento alla fine

# =====================================================================
# BLOCCO 1: OPERATORE MICROFRONTEND E PARAMETRI HARDWARE
# =====================================================================
# Importo il motore ufficiale C++ di Google. Questo mi garantisce che il PC
# calcolerà lo spettrogramma nello stesso identico modo in cui lo farà l'Arduino.
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

# Questi numeri devono coincidere millimetricamente con i parametri C++ della scheda (micro_features_micro_model_settings.h)
DATASET_PATH = 'dataset_audio_Dom_augmented' # Punta alla cartella con i file elaborati e aumentati (500 per classe)
SAMPLE_RATE = 16000 # Frequenza di campionamento fissa a 16kHz
WINDOW_SIZE_MS = 30 # Dimensione della singola finestra di ascolto (30 millisecondi)
WINDOW_STEP_MS = 20 # Di quanto scorre in avanti la finestra ogni volta (20 millisecondi)
MEL_BINS = 40       # Il numero di filtri/colonne del nostro spettrogramma

# Limiti in Hertz dei filtri passa-banda. Ripresi fedelmente dal codice Arduino. Questi sono stati fondamentali per risolvere un problema con i settaggi 
# di alcuni file della libreria tflite importata sull IDE dalla repository Github
LOWER_EDGE_HERTZ = 125.0
UPPER_EDGE_HERTZ = 7500.0


# =====================================================================
# BLOCCO 2: ESTRAZIONE DELLE CLASSI E FUNZIONI DI PREPROCESSING
# =====================================================================
# Qui definisco come un singolo file WAV crudo viene trasformato 
# in un'impronta digitale (spettrogramma) digeribile dalla rete neurale.

data_dir = pathlib.Path(DATASET_PATH) # Converto il percorso in un oggetto Path per esplorarlo facilmente

# Definisco l'array esatto e ordinato delle classi. L'ordine qui DEVE essere lo stesso dell'array C++ : altrimenti l output del codice IDE printerà
# le classi scambiate e quindi la parola sbagliata anche se sta riconoscendo gli spettrogrammi
commands = np.array(['background_noises', 'Unknown', 'clima', 'guida', 'luce', 'off', 'on', 'televisione'])
print(f"Classi rilevate ({len(commands)}): {commands}")


def decode_audio(audio_binary):
    # Decodifico i byte del file WAV e forzo la lettura a 1 singolo canale (Mono)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    # Rimuovo eventuali dimensioni inutili in eccesso dall'array per avere una lista piatta
    audio = tf.squeeze(audio, axis=-1)
    
    desired_length = 16000 # 16000 campioni(elementi che costituiscono ogni audio) che a 16000Hz corrispondono esattamente a 1 secondo
    
    # Se per qualche motivo il file è sbordato di qualche campione, lo taglio ad 1 secondo
    audio = audio[:desired_length]
    
    # Se invece il file è leggermente più corto, calcolo quanti campioni mancano
    mancanti = desired_length - tf.shape(audio)[0]
    
    # Creo una toppa di zeri (silenzio puro) lunga quanto la parte mancante
    zero_padding = tf.zeros([mancanti], dtype=tf.float32)
    # Incollo la toppa alla fine dell'audio per portarlo a 16000 campioni esatti
    audio = tf.concat([audio, zero_padding], 0)
    
    # Dico esplicitamente a TensorFlow che questo tensore ha una forma fissa e garantita
    audio.set_shape([desired_length])
    
    # Passaggio all' Hardware : Quando decode_wav legge un file audio su PC, normalizza l'onda sonora trasformandola 
    # in numeri Float32 (con la virgola) compresi tra -1.0 e +1.0. L'Arduino usa i numeri Int16 (interi), che vanno da -32768 a +32767.
    # Quindi per convertire in un formato accettato dal MicroFrontend devo moltiplicare per 32768 (che sarebbe 2 ^15). 
    audio_int16 = tf.cast(audio * 32768.0, tf.int16)
    
    return audio_int16


def get_label(file_path):
    # Estraggo la parola (label ossia la classe del mio riconoscimento vocale) dal nome della cartella in cui si trova il file
    parts = tf.strings.split(file_path, os.path.sep) # Taglio il percorso usando gli slash /
    return parts[-2] # Prendo il penultimo pezzo del percorso (es: dataset/clima/file.wav -> "clima")


def get_spectrogram(audio_int16):
    # Genero l'impronta vocale usando la libreria ufficiale Google (simulazione hardware)
    spectrogram = frontend_op.audio_microfrontend(
        audio_int16,
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE_MS,
        window_step=WINDOW_STEP_MS,
        num_channels=MEL_BINS,
        lower_band_limit=LOWER_EDGE_HERTZ,
        upper_band_limit=UPPER_EDGE_HERTZ,
        out_scale=1,
        out_type=tf.float32
    )
    
    # Divido per 25.6: per normalizzare i numeri in un range utilizzabile
    spectrogram = spectrogram / 25.6
    
    # Aggiungo una terza dimensione (canale colore) per farlo trattare come un'immagine in scala di grigi
    spectrogram = tf.expand_dims(spectrogram, -1)
    
    # Forzo le dimensioni finali dello spettrogramma: 49 finestre di tempo, 40 frequenze, 1 canale
    spectrogram.set_shape([49, 40, 1])
    
    return spectrogram


def process_path(file_path):
    # Funzione "Master" che incatena tutti i passaggi per un singolo file
    label = get_label(file_path) # Legge l'etichetta
    audio_binary = tf.io.read_file(file_path) # Legge i byte grezzi dal disco
    audio_int16 = decode_audio(audio_binary) # Pareggia la lunghezza e converte in Int16
    spectrogram = get_spectrogram(audio_int16) # Estrae lo spettrogramma
    
    # Trasformo la stringa (es. "clima") nel suo numero corrispondente (es. 2) confrontandola con l'array commands
    label_id = tf.argmax(label == commands)
    
    return spectrogram, label_id


# =====================================================================
# BLOCCO 3: CREAZIONE DELLA PIPELINE DATI (TRAIN E VALIDATION)
# =====================================================================
# Qui divido i file per l'addestramento e il test. È vitale congelare lo split
# per evitare il "Data Leakage" (imparare dai dati di test).

# Prendo la lista di tutti i file di tutte le sottocartelle
files = tf.data.Dataset.list_files(str(data_dir/'*/*'))

# Mischio la lista in modo quasi casuale perchè comunque fisso l'ordine (reshuffle_each_iteration=False).
# In questo modo, l'80% scelto per il training non si mescolerà mai col 20% di validazione nelle varie epoche.
files = files.shuffle(buffer_size=10000, seed=42, reshuffle_each_iteration=False)

num_files = files.cardinality().numpy() # Conto quanti file ci sono in totale
num_train = int(num_files * 0.8) # Calcolo la soglia per prendere esattamente l'80%

train_files = files.take(num_train) # Ritaglio i file da usare per studiare (addestramento)
val_files = files.skip(num_train)   # Ritaglio i file restanti da usare per gli esami (validazione)

AUTOTUNE = tf.data.AUTOTUNE # Dico a TensorFlow di parallelizzare e usare i processori liberi
# Applico la funzione Master a tutti i file per trasformarli da percorsi a Spettrogrammi veri e propri
train_ds = train_files.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_files.map(process_path, num_parallel_calls=AUTOTUNE)

BATCH_SIZE = 16 # Dico alla rete di analizzare 16 file alla volta prima di aggiornare i suoi pesi

# Sul set di training rimescolo l'ordine dei file ad ogni epoca (reshuffle_each_iteration=True).
# Serve a evitare che la rete impari a memoria l'ordine in cui le passo le parole.
train_ds = train_ds.shuffle(buffer_size=8000, reshuffle_each_iteration=True)
train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE) # Raggruppo a blocchi di 16 e salvo in RAM (cache) per andare più veloce

# Sul set di validazione non applico lo shuffle, mi basta passarli a blocchi di 16 per avere i risultati 
val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)


# =====================================================================
# BLOCCO 4: COSTRUZIONE DELL'ARCHITETTURA DELLA CNN
# =====================================================================
# Strutturo una Convolutional Neural Network leggerissima, studiata 
# su misura per i microcontrollori.

input_shape = (49, 40, 1) # Dimensione dell'immagine in entrata

model = models.Sequential([
    layers.Input(shape=input_shape), # Livello di ingresso
    
    # Primo blocco: estrae le caratteristiche base (linee, picchi, fruscii)
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)), # Rimpicciolisce la mappa tenendo solo i tratti più forti
    
    # Secondo blocco: capisce i pattern più complessi (le forme delle parole)
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5), # Introduco il Dropout:Spegne a caso metà dei neuroni per evitare che imparino a memoria (Overfitting)
    
    layers.Flatten(), # Schiaccia tutto in un unico vettore monodimensionale
    
    # Rete finale che decide quale parola è in base ai tratti estratti
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), # Altro giro di Dropout
    
    # Livello di uscita: 8 neuroni (le mie classi). Uso Softmax per darmi percentuali che sommano a 100
    layers.Dense(len(commands), activation='softmax')
])

model.summary() # Stampa la tabella riassuntiva dei pesi della rete

# Compilo il modello: gli do un ottimizzatore (Adam) e una bussola per calcolare l'errore
model.compile(
    # Ho abbassato il Learning Rate a 0.0005 per avere un addestramento più lento ma molto più preciso e stabile, compromesso tra 0,0001 e 0,0010
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Funzione di costo per etichette numeriche
    metrics=['accuracy'] # Voglio misurare la percentuale di risposte corrette
)


# =====================================================================
# BLOCCO 5: ADDESTRAMENTO, SALVATAGGIO E GRAFICI
# =====================================================================
# Mando in esecuzione il training e stampo i grafici finali.

EPOCHS = 100 # Numero di giri completi su tutti i file del dataset
print(f"\nInizio addestramento per {EPOCHS} epoche...")

# Faccio partire il calcolo 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Finito l'addestramento, salvo il cervello della rete neurale in formato .h5
model.save('modello_domotica.h5')
print("\nModello Keras salvato con successo come 'modello_domotica.h5'")

# Estraggo la cronologia dell'andamento (loss e accuracy) per metterli in grafico
metrics = history.history
plt.figure(figsize=(12, 5))

# Primo grafico (sinistra): Mostra se l'errore (Loss) scende correttamente
plt.subplot(1, 2, 1)
plt.plot(metrics['loss'], label='Training Loss')
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.title('Loss (Funzione di Costo)')
plt.legend()

# Secondo grafico (destra): Mostra come sale la precisione e se vado in overfitting
plt.subplot(1, 2, 2)
plt.plot(metrics['accuracy'], label='Training Accuracy')
plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
plt.title('Accuratezza')
plt.legend()

# Esporto l'immagine dei grafici nella cartella per consultarla senza tenere aperto lo script
plt.savefig('andamento_training.png')