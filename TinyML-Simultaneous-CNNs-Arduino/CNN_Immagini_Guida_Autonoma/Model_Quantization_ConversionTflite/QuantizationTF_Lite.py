import os
import random
import pathlib
import numpy as np
import tensorflow as tf
from PIL import Image

# =====================================================================
# BLOCCO 1: IMPOSTAZIONI GLOBALI
# =====================================================================
MODEL_NAME = 'modello_immagini.keras'    # Il modello appena addestrato
DATASET_PATH = 'Dataset_processed'       # La cartella con le immagini 96x96 in scala di grigi
IMG_SIZE = 96

TFLITE_FILENAME = 'modello_immagini.tflite'

print(f"[*] Avvio Quantizzazione per '{MODEL_NAME}'...")

# =====================================================================
# BLOCCO 2: CARICAMENTO DEL MODELLO E SETUP CONVERTITORE
# =====================================================================
# Carichiamo il modello Float32 dalla RAM
model = tf.keras.models.load_model(MODEL_NAME)

# Inizializziamo il convertitore TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Attiviamo l'ottimizzazione di default (necessaria per la quantizzazione)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# =====================================================================
# BLOCCO 3: GENERATORE DEL DATASET DI CALIBRAZIONE (REPRESENTATIVE DATASET)
# =====================================================================
# Raccogliamo i percorsi di tutte le immagini nel dataset processato
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
all_files = []

for root, _, files in os.walk(DATASET_PATH):
    for f in files:
        if f.lower().endswith(image_extensions):
            all_files.append(os.path.join(root, f))

# Mischiamo i file, altrimenti il convertitore 
# leggerebbe solo le prime 100 immagini  e sballerebbe 
# la calibrazione matematica per le altre classi.
random.shuffle(all_files)

def representative_dataset_gen():
    # Passiamo al convertitore 150 immagini a caso per fargli "capire" i range numerici
    for file_path in all_files[:150]:
        # Apriamo l'immagine forzandola in scala di grigi ('L')
        img = Image.open(file_path).convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convertiamo in array float32 (valori crudi 0-255). 
        # Il rescaling avverrà internamente grazie al layer aggiunto
        img_array = np.array(img, dtype=np.float32)
        
        # Aggiungiamo la dimensione del canale: (96, 96) -> (96, 96, 1)
        img_array = np.expand_dims(img_array, axis=-1)
        
        # Aggiungiamo la dimensione del batch: (96, 96, 1) -> (1, 96, 96, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        yield [img_array]

# Diciamo al convertitore di usare questa funzione
converter.representative_dataset = representative_dataset_gen

# =====================================================================
# BLOCCO 4: FULL INTEGER QUANTIZATION (FORZATURA INT8)
# =====================================================================
# Obblighiamo il sistema a usare solo operazioni compatibili con i microcontrollori 
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Forziamo i tensori di ingresso (input) e uscita (output) a essere interi a 8-bit (-128, 127).
# Questo genererà i  "Scale" e "Zero Point".
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# =====================================================================
# BLOCCO 5: ESECUZIONE E SALVATAGGIO TFLITE
# =====================================================================
print("[*] Compressione e Quantizzazione INT8 in corso ( qualche istante)...")
tflite_model = converter.convert()

# Salviamo fisicamente i byte appena generati nel file .tflite 
with open(TFLITE_FILENAME, 'wb') as f:
    f.write(tflite_model)
    
dim_kb = os.path.getsize(TFLITE_FILENAME) / 1024  #Estraggo le dimensioni del modello in .tflite
print(f"[V] Successo! Modello compresso salvato come '{TFLITE_FILENAME}' ({dim_kb:.2f} KB)")
print("\n=== QUANTIZZAZIONE COMPLETATA ===")