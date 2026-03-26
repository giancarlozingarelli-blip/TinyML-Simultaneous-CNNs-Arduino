import os

# Definiamo i nomi dei file
file_tflite = "modello_quantizzato.tflite"
file_cc = "modello_vocale_C.cc"

print(f"Generazione dell'array C++ in corso...")

# Eseguiamo il comando xxd tramite sistema 
# L'opzione -i genera l'output in formato C include
os.system(f"xxd -i {file_tflite} > {file_cc}")

if os.path.exists(file_cc):
    print(f" Successo! File '{file_cc}' generato correttamente.")
    # Stampiamo la dimensione finale per verifica
    dimensione = os.path.getsize(file_cc)
    print(f"Dimensione del file .cc: {dimensione / 1024:.2f} KB")
else:
    print(" Errore nella generazione del file.")