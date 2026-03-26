import numpy as np # Importo numpy per fare i conti veloci sugli array e trovare la predizione massima
import tensorflow as tf # Il motore principale
# Importo il modulo C++ di Google per assicurarmi che lo spettrogramma generato qui sia 
# identico al 100% a quello che calcolerà la libreria su Arduino
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op

# =====================================================================
# BLOCCO 1: IMPOSTAZIONI DEL TEST
# =====================================================================
# Qui decido quale file utilizzare come test e carico i due modelli da confrontare.

# Cambio questa stringa ogni volta che voglio testare un file diverso per vedere come reagiscono i modelli
AUDIO_TEST_FILE = "dataset_audio_Dom_processed/luce/luce_050.wav" 

MODEL_H5 = "modello_domotica.h5" # Il modello originale, pesante e preciso (calcoli con la virgola)
MODEL_TFLITE = "modello_quantizzato.tflite" # Il modello compresso per Arduino (calcoli a numeri interi)

# L'ordine delle classi deve essere lo stesso del training e del file C++
COMMANDS = ['background_noises', 'Unknown', 'clima', 'guida', 'luce', 'off', 'on', 'televisione']

# =====================================================================
# BLOCCO 2: FUNZIONE DI PREPROCESSING 
# =====================================================================
# Prendo l'audio grezzo e lo trasformo in cio che si aspetta la rete.

def preprocess_audio(file_path):
    print(f" Caricamento audio: {file_path}")
    
    audio_binary = tf.io.read_file(file_path) # Leggo fisicamente i byte del file WAV
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1) # Decodifico in Mono
    audio = tf.squeeze(audio, axis=-1) # Appiattisco l'array per togliere dimensioni inutili
    
    # Forzo  la lunghezza a 1 secondo esatto (16000 campioni)
    desired_length = 16000
    audio = audio[:desired_length] # Taglio se è troppo lungo
    
    # Se è troppo corto, conto quanto manca e lo riempio di zeri (silenzio) alla fine
    mancanti = desired_length - tf.shape(audio)[0]
    zero_padding = tf.zeros([mancanti], dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([desired_length]) # mi assicuro che la lunghezza è 16000
    
    # Simulo il microfono dell'Arduino convertendo i valori decimali in interi a 16 bit
    audio_int16 = tf.cast(audio * 32768.0, tf.int16)
    
    # Genero lo spettrogramma passando i numeri interi al MicroFrontend C++
    spectrogram = frontend_op.audio_microfrontend(
        audio_int16, sample_rate=16000, window_size=30, window_step=20,
        num_channels=40, lower_band_limit=125.0, upper_band_limit=7500.0,
        out_scale=1, out_type=tf.float32
    )
    
    spectrogram = spectrogram / 25.6 # Divisione di normalizzazione richiesta da TensorFlow Lite
    spectrogram = tf.expand_dims(spectrogram, -1) # Aggiungo il canale per simulare un'immagine in scala di grigi
    
    # Aggiungo un'ulteriore dimensione all'inizio (Batch dimension). 
    # La rete neurale vuole sempre una "pila" di immagini, quindi  do una pila con 1 immagine
    spectrogram = tf.expand_dims(spectrogram, 0) 
    
    return spectrogram


# =====================================================================
# BLOCCO 3: TEST SUL MODELLO KERAS ORIGINALE
# =====================================================================
# Metto alla prova il modello grande che ragiona con i precisi numeri Float32.

print("\n" + "="*50)
print(" TEST 1: Modello Keras Originale (Float32)")
print("="*50)

# Trasformo il mio file WAV di test in uno spettrogramma
spectrogram_input = preprocess_audio(AUDIO_TEST_FILE)

model_keras = tf.keras.models.load_model(MODEL_H5) # Carico il modello pesante in RAM
# Faccio fare la predizione. verbose=0 spegne i messaggi fastidiosi di log.
# Prendo l'elemento [0] perché il risultato è incapsulato dentro un array di batch.
pred_keras = model_keras.predict(spectrogram_input, verbose=0)[0]

print("Probabilità Keras:")
# Scorro l'array delle probabilità e le stampo affiancate al nome del comando
for i, cmd in enumerate(COMMANDS):
    # Moltiplico per 100 per farla sembrare una percentuale umana (es: 99.5%)
    print(f" - {cmd.ljust(18)}: {pred_keras[i]*100:>6.2f}%")


# =====================================================================
# BLOCCO 4: TEST SUL MODELLO TFLITE (LA SIMULAZIONE HARDWARE)
# =====================================================================
# Simulo riga per riga quello che succederà dentro il chip dell'Arduino Nano,
# calcolando tutto a numeri interi a 8-bit.

print("\n" + "="*50)
print(" TEST 2: Modello Quantizzato TFLite (INT8)")
print("="*50)

# Avvio l'interprete leggero di TensorFlow Lite e gli passo il file compresso
interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors() # Riservo la memoria RAM esatta (Tensor Arena) che userà la rete

# Estraggo la mappa di dove la rete si aspetta i dati in ingresso e dove sputerà quelli in uscita
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Catturo i parametri di calibrazione (Scale e Zero Point) dal modello quantizzato.
# Sono gli stessi numeri da mettere nel codice C++
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

 
# Prendo lo spettrogramma perfetto (Float32) e lo schiaccio forzatamente in Int8 usando la formula di quantizzazione.
input_data_int8 = spectrogram_input / input_scale + input_zero_point
input_data_int8 = tf.cast(tf.round(input_data_int8), tf.int8) # Arrotondo e converto materialmente in 8-bit

# Inietto i dati schiacciati dentro l'imbuto di ingresso della rete
interpreter.set_tensor(input_details[0]['index'], input_data_int8)
interpreter.invoke() # Faccio partire  l'inferenza (i calcoli neurali)

# Vado a prendere il risultato grezzo dall' uscita (saranno dei numeri interi da -128 a 127)
output_data_int8 = interpreter.get_tensor(output_details[0]['index'])[0]

# Prendo i numeri grezzi a 8-bit e applico la formula inversa (Dequantizzazione) 
# per trasformarli di nuovo in probabilità decimali comprensibili a me.
pred_tflite = (output_data_int8.astype(np.float32) - output_zero_point) * output_scale

print("Probabilità TFLite INT8:")
for i, cmd in enumerate(COMMANDS):
    # Stampo  i risultati dell'Arduino simulato
    print(f" - {cmd.ljust(18)}: {pred_tflite[i]*100:>6.2f}%")


# =====================================================================
# BLOCCO 5: DIAGNOSI FINALE 
# =====================================================================
# Confronto i due modelli per vedere se la compressione a 8-bit ha distrutto l'intelligenza della rete.

print("\n" + "="*50)
print(" DIAGNOSI DEL DEBUG:")

# np.argmax trova la posizione (indice) del numero più alto nell'array. 
# Lo uso per capire quale classe ha "vinto"(ossia quae parola il modello ha predetto).
winner_keras = COMMANDS[np.argmax(pred_keras)]
winner_tflite = COMMANDS[np.argmax(pred_tflite)]

# Stampo la predizione di Keras e di TFLite
print(f"Predizione Keras:  {winner_keras} ({np.max(pred_keras)*100:.1f}%)")
print(f"Predizione TFLite: {winner_tflite} ({np.max(pred_tflite)*100:.1f}%)")

# La prova del nove finale:
if winner_keras != winner_tflite:
    # Se i due modelli danno una parola diversa per lo stesso file audio, significa che schiacciare 
    # i numeri a 8 bit ha confuso la rete. La quantizzazione ha fallito.
    print(" ATTENZIONE: La quantizzazione INT8 ha corrotto il modello, Le predizioni non coincidono.")
else:
    # Se dicono la stessa cosa, la mia pipeline Python è una roccia e il modello TFLite è sano.
    # Da questo momento, qualsiasi errore dal vivo è puramente un problema hardware o C++.
    print(" OK: Il modello TFLite ragiona esattamente come il modello Keras.")
    print("Se su Arduino fallisce, il problema è al 100% nel codice C++ (generazione spettrogramma o lettura output).")