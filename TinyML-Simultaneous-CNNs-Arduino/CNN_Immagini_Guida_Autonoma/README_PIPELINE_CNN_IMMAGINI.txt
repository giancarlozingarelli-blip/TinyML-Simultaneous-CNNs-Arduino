PIPELINE CNN_IMMAGINI:

1)Scaricare il dataset

1*)Qui si consiglia come per lo sviluppo dell'altra CNN, di creare un ambiente di sviluppo WSL e venv , nello stesso modo dell' alto caso ma sfruttando il file requirements.txt insieme a questo README.TXT. Va scaricato, importato nella cartella di progetto wsl e una volta aperto il venv bisogna eseguire: pip install -r requirements.txt . In questo modo ho reinstallato tutti i pacchetti dell' altro ambiente a cui andranno aggiunti altri , dopo elencati.

2)Eseguire lo script di Preprocessing

3)Eseguire lo script di Training

4)Eseguire lo script di Conversione e Quantizzazione in formato .tflite

5)Eseguire lo script di Conversione in array C++

6)Installare sull' IDE la libreria nativa Arduino_OV767X by Arduino

7)Aggiungere allo Sketch Ide aperto per la CNN vocale , Il nostro modello CNN per la guida autonoma trasformato in array C++ nel punto 5; avendo cura di fargli le stesse modifiche fatte al modello Vocale C++ quando fu portato sull' IDE. Stesso discorso per la creazione del suo file header.

8) Cancellare completamente il contenuto del file .ino nello sketch ed incollare il nuovo file .ino trovato nella cartella Ambiente_Arduino_COmpleto_IDE. Aperta la cartella, ci sarà un' altra cartella, ed al suo interno il file .ino chiamato SCRIPT_IDE_COMPLETO.ino . 

9)Caricare lo sketch sul microcontrollore


Requisiti di sistema:

Sarà necessario integrare con i seguenti pacchetti : pip install pyserial Pillow.


