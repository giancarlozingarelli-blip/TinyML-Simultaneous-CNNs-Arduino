import tensorflow as tf # Il motore principale
import numpy as np # Per le operazioni matematiche di base
import pathlib # Per navigare facilmente tra le cartelle
import random # Fondamentale per mischiare i file e non sballare la calibrazione

# =====================================================================
# BLOCCO 1: MOTORE MICROFRONTEND
# =====================================================================
# Importo di nuovo l'operatore C++ di Google. Anche in fase di conversione, 
# dobbiamo processare l'audio nello stesso identico modo del training.
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

# =====================================================================
# BLOCCO 2: PARAMETRI HARDWARE E FILE
# =====================================================================
# Questi valori sono scolpiti nella pietra. Se ne cambio uno qui, il modello 
# quantizzato genererà parametri Scale e Zero Point sballati per l'Arduino.
SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 30
WINDOW_STEP_MS = 20
MEL_BINS = 40
LOWER_EDGE_HERTZ = 125.0
UPPER_EDGE_HERTZ = 7500.0

# Il nome del file Keras (Float32) che abbiamo appena creato
MODEL_NAME = 'modello_domotica.h5' 
# La cartella da cui pescheremo i file per fare la calibrazione
DATASET_PATH = 'dataset_audio_Dom_augmented' 

# =====================================================================
# BLOCCO 3: CARICAMENTO DEL MODELLO E SETUP CONVERTITORE
# =====================================================================
print(f"Caricamento del modello '{MODEL_NAME}' in corso...")
# Prendo il modello  in RAM
model = tf.keras.models.load_model(MODEL_NAME)
# Inizializzo lo strumento di TensorFlow che compressa i modelli per i microcontrollori
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Dico al convertitore di usare le ottimizzazioni di base (inizia a prepararsi a ridurre la precisione)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# =====================================================================
# BLOCCO 4: PREPARAZIONE DEL DATASET DI CALIBRAZIONE
# =====================================================================
# Per convertire la matematica da Float32 a Int8 (-128 a 127), il convertitore
# deve "ascoltare" un po' di audio reale per capire quanto sono grandi i numeri in media.

data_dir = pathlib.Path(DATASET_PATH)
# Pesco la lista completa di tutti i file WAV nel dataset
files = tf.io.gfile.glob(str(data_dir) + '/*/*.wav')

# Mescolo i file a caso, altrimenti TensorFlow pesca i primi 100 file in ordine alfabetico (quindi solo "background_noises").
# Il convertitore si calibrerebbe solo sul rumore,mandando i numeri fuori scala qualsiasi parola viene detta al microfdono arduino.
random.shuffle(files)

def decode_audio(audio_binary):
    # Stessa  funzione usata nel training: prende il WAV grezzo e lo forza a 1 secondo, mono, 16000Hz.
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    desired_length = 16000
    audio = audio[:desired_length]
    mancanti = desired_length - tf.shape(audio)[0]
    zero_padding = tf.zeros([mancanti], dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([desired_length])
    
    # Moltiplico per 32768 per simulare il microfono hardware che dà interi a 16 bit
    audio_int16 = tf.cast(audio * 32768.0, tf.int16)
    return audio_int16

def get_spectrogram(audio_int16):
    # Stesso calcolo dello spettrogramma fatto nel training. 
    # Assicura che l'impronta vocale data al convertitore sia geometricamente perfetta.
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
    # Stessa calibrazione su TFLite
    spectrogram = spectrogram / 25.6
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram.set_shape((49, 40, 1))
    return spectrogram


# =====================================================================
# BLOCCO 5: GENERATORE E REGOLE DI QUANTIZZAZIONE (INT8)
# =====================================================================

def representative_dataset_gen():
    # Questa funzione è un "generatore". Fornisce al convertitore un file audio
    # processato alla volta, così non intasa la RAM del PC.
    
    # Decido di fargli ascoltare massimo 100 file (più che sufficienti per calibrare i pesi)
    num_calibration_steps = min(100, len(files))
    
    for i in range(num_calibration_steps):
        audio_binary = tf.io.read_file(files[i]) # Legge i byte dal disco
        audio_int16 = decode_audio(audio_binary) # Converte in PCM 16-bit
        spectrogram = get_spectrogram(audio_int16) # Estrae lo spettrogramma
        
        # 'yield' è come un return, ma non ferma la funzione.
        # tf.expand_dims(..., 0) serve ad aggiungere una dimensione vuota all'inizio 
        # perché la rete neurale accetta sempre pacchetti (batch), quindi fingo un batch da 1.
        yield [tf.expand_dims(spectrogram, 0)]

# Dico al convertitore di usare il generatore appena creato per calibrare i numeri
converter.representative_dataset = representative_dataset_gen

# Forzo tutte le operazioni matematiche interne a usare interi a 8-bit.
# L'Arduino Nano 33 BLE andrebbe lentissimo a calcolare numeri con la virgola (Float).
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Obbligo il modello ad accettare input a 8-bit (-128, 127) e a sputare output a 8-bit.
# Da ottengo le variabili g_input_scale e g_input_zero_point del nostro script C++
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8 


# =====================================================================
# BLOCCO 6: ESECUZIONE E SALVATAGGIO SU DISCO
# =====================================================================

print("Inizio quantizzazione (potrebbe richiedere qualche minuto)...")
# Eseguo materialmente la compressione e il ricalcolo di tutti i pesi della rete
tflite_model = converter.convert()

tflite_filename = 'modello_quantizzato.tflite'
# Apro un nuovo file in modalità scrittura binaria ('wb')
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model) # Metto i byte appena generati nel file fisico

print(f"\nSuccesso! Il modello compresso è stato salvato come '{tflite_filename}'")