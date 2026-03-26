import tensorflow as tf # motore principale di intelligenza artificiale  Google
from tensorflow import keras # interfaccia semplice e ad alto livello per costruire la rete neurale
from tensorflow.keras import layers # layer con cui costruiremo l'architettura del modello
import matplotlib.pyplot as plt # libreria grafica per disegnare l'andamento dell'addestramento alla fine
import numpy as np #  libreria fondamentale per la matematica complessa e la gestione degli array

# =====================================================================
# BLOCCO 1: IMPOSTAZIONI GLOBALI
# =====================================================================
DATASET_DIR = 'Dataset_processed' # La cartella con le proprie immagini 96x96 in grigio, generate dal preprocessing
IMG_SIZE = 96 # Dimensione hardware-target fissa: 96x96 pixel per rispettare la poca RAM della scheda
BATCH_SIZE = 16 # Quante immagini la rete guarda contemporaneamente prima di aggiornare la sua memoria 

print(f"TensorFlow Version: {tf.__version__}") # Stampo la versione di TF 
print("Inizio la pipeline dei dati a 1 canale con cambio marcia automatico...") # Avviso a schermo dell'avvio

# =====================================================================
# BLOCCO 2: CARICAMENTO DATI 
# =====================================================================
# Carico le immagini per il training (80% del totale). color_mode="grayscale" forza un solo canale (meno dati, meno RAM occupata)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123,
    color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
)

# Carico il restante 20% che userò per gli esami e i test finali. Il seed=123 garantisce che lo spacco 80/20 sia identico al blocco sopra
val_test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
    color_mode="grayscale", image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
)

class_names = train_ds.class_names # Estraggo automaticamente i nomi delle classi dai nomi delle cartelle
print(f"Classi rilevate: {class_names}") # Le stampo per conferma

# Dividiamo il 20% rimanente a metà esatta: 10% per la Validazione (esami intermedi), 10% Test (esame di stato finale)
val_batches = len(val_test_ds) # Conto quanti "pacchetti" da 16 ci sono nel 20%
test_ds = val_test_ds.take(val_batches // 2) # Prendo la prima metà per il test set finale
val_ds = val_test_ds.skip(val_batches // 2) # Salto la prima metà e prendo la seconda per la validazione continua

# =====================================================================
# BLOCCO 3: DATA AUGMENTATION (SICURA IN RAM)
# =====================================================================
# Creo un mini-modello che "sporca" le immagini in tempo reale per evitare che la rete impari a memoria (Overfitting)
data_augmentation = keras.Sequential([
    keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1)), # Dichiaro la forma: 96x96, 1 solo colore (grigio)
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=0.0), # Sposta l'immagine su/giù/dx/sx del 10% riempiendo i buchi di nero
    layers.RandomZoom(height_factor=0.1, fill_mode='constant', fill_value=0.0), # Zooma in o out del 10%
    layers.RandomFlip("horizontal") # Specchia l'immagine orizzontalmente (utile se un ostacolo può apparire speculare)
])

AUTOTUNE = tf.data.AUTOTUNE # Dico a TensorFlow di gestire da solo quanta CPU usare per caricare i dati velocemente

# Preparo il training: metto in cache, mischio pesantemente (1000), applico l'augmentation e preparo il blocco successivo (prefetch)
train_ds = train_ds.cache().shuffle(1000).map(
    lambda x, y: (data_augmentation(x, training=True), y), # Applica le distorsioni solo ai dati x (le immagini), non alle y (le etichette)
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# Per validazione e test non distorco nulla, Devono essere immagini pulite e reali. Li metto solo in cache per maggiore velocità.
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# =====================================================================
# BLOCCO 4: ARCHITETTURA ULTRA-TINY V2 (SEPARABLE CONVOLUTIONS)
# =====================================================================
print("\n[!] Creazione della Ultra-Tiny CNN V2 (Separable)...")

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1)) # Punto d'ingresso: il tensore dell'immagine grezza

# 1. Rescaling integrato: trasforma i pixel da 0-255 a 0.0-1.0. Fondamentale per far capire i numeri alla rete
x = layers.Rescaling(1./255)(inputs)

# 2. PRIMO BLOCCO (Conv standard, dimezza subito l'immagine per salvare RAM)
# strides=2 salta un pixel ogni due, dimezzando immediatamente l'immagine a 48x48: Salva molta RAM
x = layers.Conv2D(16, (3, 3), strides=2, padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x) # Normalizza i dati in uscita per rendere l'addestramento super stabile
x = layers.Activation('relu')(x) # Funzione di attivazione che accende i neuroni solo se il valore è positivo

# 3. SECONDO BLOCCO (Separable: estrae forme complesse con pochi calcoli)
# SeparableConv2D fa la stessa cosa del Conv2D ma con una frazione delle moltiplicazioni matematiche. Perfetto per i chip poco potenti!
x = layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2))(x) # Rimpicciolisce ancora l'immagine tenendo solo i segnali più forti (es. bordi netti)

# 4. TERZO BLOCCO: Aumento i filtri a 64 per cercare pattern più complessi (forme intere di ostacoli)
x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# 5. QUARTO BLOCCO (Spingiamo l'intelligenza a 128 filtri!) Livello massimo di astrazione.
x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# 6. CLASSIFICATION HEAD: Trasformiamo le mappe spaziali in un vettore di probabilità
# Sostituisce il Flatten(): fa la media di tutte le mappe estraendo un solo numero per mappa. Risparmia tantissimi parametri (Peso in KB)
x = layers.GlobalAveragePooling2D()(x) 
x = layers.Dropout(0.5)(x) # Spegne il 50% dei neuroni a caso ad ogni giro per costringere gli altri a imparare meglio (anti-overfitting)

# Livello finale denso: tanti neuroni quante sono le nostre classi. Softmax trasforma i valori in percentuali che sommano a 100%
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs) # Assemblo fisicamente il modello collegando l'ingresso all'uscita
model.summary() # Stampo la tabella con il conto totale dei parametri (fondamentale per capire se entra nella memoria Flash di Arduino)

# =====================================================================
# BLOCCO 5: ADDESTRAMENTO AUTOMATICO
# =====================================================================
print("\n=======================================================")
print(" ADDESTRAMENTO DA ZERO AUTOMATICO ATTIVATO)")
print("=======================================================")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), # Partiamo con una velocità normale (passo di apprendimento)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Il giudice che valuta quanto la rete sta sbagliando
    metrics=['accuracy'] # Vogliamo leggere a schermo la percentuale di corrette
)

# Se la validation loss non migliora per 5 epoche, dimezza il Learning Rate
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', # Osserva l'errore sui dati di esame (validazione)
    factor=0.5,        # Dimezza la velocità di apprendimento per fare passi più corti e precisi
    patience=5,        # Aspetta 5 epoche a vuoto prima di intervenire
    min_lr=0.00001,    # Non scende mai sotto questa soglia di velocità
    verbose=1          # Ti scriverà nel terminale "Epoch X: ReduceLROnPlateau reducing learning rate..."
)

#Si ferma da solo e ti ridà la versione migliore se l'addestramento va in stallo prolungato, così non posso rischiare di non aver usato abbastanza epoche
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', # Osserva sempre l'errore di validazione
    patience=15,       # Ha 15 epoche di tempo per sbloccarsi prima di arrendersi 
    restore_best_weights=True # Quando si ferma, butta via gli ultimi tentativi e ricarica i pesi dell'epoca migliore in assoluto
)

# Mettiamo un limite alto, l'EarlyStopping interverrà prima.
EPOCHS = 100 

# Lancio l'addestramento  assegnandolo a una variabile "history" per poterne poi disegnare i grafici
history = model.fit(
    train_ds, # I dati su cui studiare
    validation_data=val_ds, # I dati su cui fare gli esami intermedi
    epochs=EPOCHS, # Il limite massimo di giri
    callbacks=[early_stop, lr_scheduler] # Passiamo entrambi gli "aiutanti" (callbacks) alla rete
)

# =====================================================================
# BLOCCO 6: VALUTAZIONE E SALVATAGGIO
# =====================================================================
print("\nValutazione finale su Test Set (Dati MAI VISTI prima dal modello):")
test_loss, test_acc = model.evaluate(test_ds) # Esame di stato finale per capire la vera affidabilità 
print(f"Accuratezza Test Set: {test_acc*100:.2f}%\n")

model.save('modello_immagini.keras') # Salvo il modello addestrato nel nuovo formato compatto di Keras
print("Modello salvato con successo come 'modello_immagini.keras'")

# =====================================================================
# BLOCCO 7: GRAFICI
# =====================================================================
# Estraggo i dati storici salvati durante l'addestramento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8)) # Preparo una tela bianca 8x8 per i grafici
plt.subplot(2, 1, 1) # Divido la tela in due righe, posizionandomi sulla riga in alto
plt.plot(acc, label='Training Accuracy') # Disegno la linea dell'accuratezza in fase di studio
plt.plot(val_acc, label='Validation Accuracy') # Disegno la linea dell'accuratezza in fase di esame
plt.legend(loc='lower right') # Metto la legenda in basso a destra
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2) # Mi posiziono sulla riga in basso
plt.plot(loss, label='Training Loss') # discesa dell'errore di studio
plt.plot(val_loss, label='Validation Loss') #  discesa dell'errore di esame
plt.legend(loc='upper right') # la legenda in alto a destra
plt.title('Training and Validation Loss')
plt.xlabel('epoch') #  l'asse delle X sarà delle "epoche"

plt.savefig('andamento_training_immagini.png') # Esporto il  file immagine PNG
print("Grafici salvati in 'andamento_training_immagini.png'")