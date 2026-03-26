PIPELINE CNN_VOCALE: 

1)Scaricare il dataset

1*)Si consiglia di creare un ambiente venv su WSL ed eseguirlo (Per crearlo : python3 -m venv venv  ;  Per attivarlo: source venv/bin/activate)

2)Eseguire lo script di Preprocessing dei dati

3)Eseguire lo script di Data Augmentation

4)Eseguire lo script di Training

5)Eseguire lo script di Conversione_Con_Quatizzazione in formato TFlite

5*) Eseguire lo script di Verifica/Debugging del funzionamento dei due modeli in formato .h5 e in formato  .tflite . 

6)Eseguire lo script di Conversione ad array C++ (giusto per comodità , in alternativa usare solo il comando a riga di comando xxd)

7)Scaricare tutti i file IDE ed aprirli in un solo sketch Arduino IDE

8)Caricare il modello sul microcontrollore tramite cavo usb e fraccia in alto a sinistra su Arduino IDE 


Requisiti di sistema: 

1)Uso dell' estensione WSL UBUNTU su Visual Studio ; 2)Attivazione velocizzazione del modello di training tramite verifica di utilizzo della GPU della scheda grafica ; 3) I seguenti pacchetti :
numpy, matplotlib, pydub, tensorflow 2.15, ffmpeg (sistema), CUDA 11.8, cuDNN 8.6, driver NVIDIA aggiornati. 
