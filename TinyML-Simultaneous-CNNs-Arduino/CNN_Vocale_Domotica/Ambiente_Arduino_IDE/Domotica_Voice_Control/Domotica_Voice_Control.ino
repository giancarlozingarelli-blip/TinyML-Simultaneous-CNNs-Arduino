// =====================================================================
// BLOCCO 1: LIBRERIE E IMPORTAZIONI
// =====================================================================
// Qui chiamo in causa tutti i file di sistema e le librerie di TensorFlow
// che mi servono per far funzionare la rete neurale sulla scheda.

#include <TensorFlowLite.h> // Includo la libreria madre di TensorFlow Lite per microcontrollori
#include "modello_vocale_C.h" // Importo l'array  generato in Python che contiene fisicamente i pesi della mia rete
#include "audio_provider.h" // Importo lo script che gestisce l'accensione e lo spegnimento del microfono hardware
#include "feature_provider.h" // Importo lo script che traduce i campioni audio grezzi in immagini (spettrogrammi)
#include "micro_features_micro_model_settings.h" // Importo le regole fisiche del progetto (frequenza 16kHz, canali, finestre)
#include "recognize_commands.h" // Importo l'algoritmo che fa la media dei risultati per evitare i falsi allarmi
#include "tensorflow/lite/micro/all_ops_resolver.h" // Importo la mappa di tutte le operazioni matematiche (addizioni, convoluzioni) supportate
#include "tensorflow/lite/micro/micro_interpreter.h" // Importo il "cervello" esecutivo che prenderà i dati e li farà passare nella rete
#include "tensorflow/lite/micro/micro_log.h" // Importo la funzione speciale MicroPrintf per stampare testi complessi sulla Seriale
#include "tensorflow/lite/micro/system_setup.h" // Importo i comandi a basso livello per inizializzare il processore dell'Arduino Nano
#include "tensorflow/lite/schema/schema_generated.h" // Importo la struttura dati standard che TensorFlow usa per leggere i file .tflite

// =====================================================================
// BLOCCO 2: VARIABILI GLOBALI E MEMORIA
// =====================================================================
// Preparo lo spazio nella memoria RAM dell'Arduino. Dichiaro i puntatori 
// partendo da "vuoto" (nullptr), li riempirò dopo nel setup.

const tflite::Model* model = nullptr; // Creo un puntatore per ospitare la struttura del modello AI
tflite::MicroInterpreter* interpreter = nullptr; // Creo il puntatore per il coordinatore (l'interprete)
TfLiteTensor* input = nullptr; // Creo il puntatore per l'"imbuto" dove butterò lo spettrogramma
TfLiteTensor* output = nullptr; // Creo il puntatore per l'"imbuto" da cui usciranno le probabilità finali

constexpr int kTensorArenaSize = 60 * 1024; // Definisco una costante: voglio rubare esattamente 60 Kilobyte di RAM
alignas(16) uint8_t tensor_arena[kTensorArenaSize]; // Creo fisicamente l'array di 60KB in RAM, allineandolo a blocchi di 16 byte per far scorrere i dati più velocemente

FeatureProvider* feature_provider = nullptr; // Puntatore al traduttore di spettrogrammi
int32_t previous_time = 0; // Variabile per ricordarmi a che millisecondo ero rimasto all'ultimo giro di loop
RecognizeCommands* recognizer = nullptr; // Puntatore al "giudice" che valuta lo storico delle predizioni

// --- NUOVE VARIABILI GLOBALI PER LA QUANTIZZAZIONE ---
float g_input_scale = 0.0f; // Variabile globale per salvare il fattore di scala (quanto devo allargare i numeri INT8)
int g_input_zero_point = 0; // Variabile globale per salvare lo "zero point" (qual è il valore del silenzio assoluto in INT8)
// -----------------------------------------------------

// Definisco l'array di testi con i nomi delle mie classi. Per l' ordine bisogna rispettare quello dell'addestramento Python.
const char* labels[kCategoryCount] = {
  "background_noises", // L'indice 0 corrisponde alla classe del rumore
  "Unknown",           // L'indice 1 corrisponde alle parole scartate
  "clima",             // L'indice 2 corrisponde al comando clima
  "guida",             // L'indice 3 corrisponde al comando guida
  "luce",              // L'indice 4 corrisponde al comando luce
  "off",               // L'indice 5 corrisponde al comando off
  "on",                // L'indice 6 corrisponde al comando on
  "televisione"        // L'indice 7 corrisponde al comando televisione
};

// =====================================================================
// BLOCCO 3: SETUP (AVVIO DELLA SCHEDA E DEL MODELLO)
// =====================================================================
// Funzione standard di Arduino che gira una volta sola quando collego la scheda al PC via usb.

void setup() {
  Serial.begin(115200); // Accendo la comunicazione USB/Seriale impostando una velocità di 115200 baud. Sul serial monitor poi andrà necessariamente inserita la stessa velocità (115200 baud)
  while (!Serial); // Metto in pausa il codice: non va avanti finché non apro la finestra del Monitor Seriale sul PC
  
  tflite::InitializeTarget(); // Mando un comando a basso livello per settare i clock e i pin della scheda Nano 33 BLE
  MicroPrintf("Avvio inizializzazione TFLM..."); // Stampo un messaggio per confermare che l'hardware è pronto
  
  model = tflite::GetModel(modello_quantizzato_tflite); // Vado a leggere l'array di byte C++ del mio modello e lo aggancio al puntatore "model"
  
  // Controllo di sicurezza: verifico se il modello è stato compilato con una versione compatibile con la libreria caricata
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Errore: Schema modello non compatibile!"); // Se non combaciano, avviso l'utente
    return; // Fermo l'esecuzione del setup (la scheda non farà più niente)
  }
  
  static tflite::AllOpsResolver resolver; // Carico in memoria TUTTE le operazioni matematiche di TensorFlow 
  
  // Costruisco fisicamente l'interprete passandogli il modello, la matematica, lo spazio in RAM e la grandezza dello spazio
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter; // Aggancio l'interprete creato al mio puntatore globale
  
  // Ordino all'interprete di prendersi possesso dei 60KB di RAM per organizzare i suoi pesi e neuroni
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("Errore: Memoria Tensor Arena insufficiente!"); // Se 60KB non gli bastano, me lo dice
    return; // E blocco di nuovo tutto
  }
  
  input = interpreter->input(0); // Collego il puntatore "input" all'esatto indirizzo di memoria dove la rete si aspetta i dati in entrata
  output = interpreter->output(0); // Collego il puntatore "output" al punto in cui la rete scriverà le sue probabilità finali

  // --- SALVATAGGIO PARAMETRI DAL MODELLO ---
  g_input_scale = input->params.scale; // Estraggo dalle impostazioni del modello la variabile Scale per la quantizzazione
  g_input_zero_point = input->params.zero_point; // Estraggo dalle impostazioni del modello il valore Zero Point per il silenzio
  // -----------------------------------------
  
  // Inizializzo il costruttore di spettrogrammi e gli dico di scrivere i risultati direttamente nel buffer di input della rete
  static FeatureProvider static_feature_provider(kFeatureElementCount, input->data.int8);
  feature_provider = &static_feature_provider; // Lo aggancio al puntatore globale
  
  // Imposto le regole per il post-processing: 
  // 1000: finestra di 1 sec in cui fa la media dei risultati;
  // 240: soglia altissima (modificabile a piacere, indica il grado di sicurezza richiesto, es. 240/255);
  // 1000: pausa di 1s tra comandi (quanto in fretta la scheda torna ad ascoltare dopo aver eseguito un comando);
  // 2: hit consecutivi richiesti (la rete deve confermare la parola per almeno 2 cicli di fila. 
  // Più hit = minor rischio di falsi positivi, ma richiede di pronunciare la parola più lentamente).
  static RecognizeCommands static_recognizer(1000, 240, 1000, 2);static RecognizeCommands static_recognizer(1000, 240, 1000, 2);
  recognizer = &static_recognizer; // Aggancio il "giudice" al puntatore globale
  
  // Tento di accendere l'hardware del microfono PDM
  if (InitAudioRecording() != kTfLiteOk) {
    MicroPrintf("Errore: Impossibile inizializzare il microfono!"); // Se il microfono è guasto o i pin sono sbagliati
    return; // Blocco tutto
  }
  MicroPrintf("Microfono attivato! In ascolto..."); // Conferma che è tutto apposto e il microfono funziona
}

// =====================================================================
// BLOCCO 4: LOOP INFINITO (ASCOLTO E INFERENZA)
// =====================================================================
// Funzione nativa di Arduino che gira in continuazione, migliaia di volte al secondo.

void loop() {
  const int32_t current_time = LatestAudioTimestamp(); // Chiedo all'orologio del microfono a che millisecondo assoluto siamo arrivati
  int how_many_new_slices = 0; // Inizializzo un contatore a zero per capire quante "fette" di audio da 30ms ha raccolto il microfono
  
  // Prendo il suono crudo dal microfono e lo faccio trasformare in spettrogramma
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      previous_time, current_time, &how_many_new_slices);
      
  // Se la conversione matematica del suono fallisce per qualche motivo
  if (feature_status != kTfLiteOk) {
    MicroPrintf("Errore nella generazione delle features audio"); // Stampo l'errore
    return; // Interrompo solo questo specifico giro di loop e riparto dall'inizio
  }
  
  // Aggiorno la variabile del tempo aggiungendo i millisecondi delle fette processate (numero fette * 20ms di salto)
  previous_time += how_many_new_slices * kFeatureSliceStrideMs;
  
  // Se il loop ha girato così velocemente che il microfono non ha ancora nulla di nuovo da darci
  if (how_many_new_slices == 0) {
    return; // Salto il giro per non far impazzire la rete con gli stessi dati di prima
  }
  
  // =========================================================
  // DEBUGGING: STAMPA I DATI GREZZI (Solo per i primi 3 cicli)
  // =========================================================
  static int debug_count = 0; // Creo un contatore statico che non si azzera tra un loop e l'altro
  if (debug_count < 3) { // Entro qui dentro solo per le primissime 3 volte dall'accensione
      int scale_int = g_input_scale * 100000; // Moltiplico il float Scale per 100.000 perché in C++ stampare i decimali è rognoso
      Serial.print(" DEBUGGING - Scale: 0."); Serial.print(scale_int); // Stampo per vedere il valore di Scale
      Serial.print(" | Zero Point: "); Serial.println(g_input_zero_point); // Stampo il parametro Zero Point affianco
      
      Serial.print("Dati Audio: "); // Inizio a stampare l'array in ingresso
      for(int i = 0; i < 15; i++) { // Scorro solo i primissimi 15 valori (sui 1960 totali) per non bloccare la Seriale
          Serial.print((int)input->data.int8[i]); // Converto il byte a 8-bit in un intero leggibile e lo stampo
          Serial.print(" "); // Aggiungo uno spazio per separare i numeri
      }
      Serial.println("\n-----------------------------------------"); // Stampo una riga di decoro per chiudere  il Debugging
      debug_count++; // Aumento il contatore di 1 (arrivato a 3, questo blocco morirà per sempre)
  }
  // =========================================================

  // =========================================================
  // FILTRO V.A.D. (Voice Activity Detection) : Lo introduco perchè avevo troppi falsi positivi , quindi imposto questa 'barriera' contro i suoni rilevati
  // =========================================================
  int max_energy = 0; // Inizializzo a zero la variabile che memorizzerà il picco massimo di volume
  for (int i = 0; i < 1960; i++) { // Apro un ciclo FOR per scorrere letteralmente tutti i 1960 pixel del mio spettrogramma(49 finestre x 40 frequenze=1960 )
      int current_val = input->data.int8[i]; // Prendo il valore del pixel attuale (che va da -128 a +127)
      int energy = abs(current_val - g_input_zero_point); // Calcolo l'energia assoluta togliendo la base del silenzio (-128)
      if (energy > max_energy) { // Se l'energia attuale batte il record precedente
          max_energy = energy; // Salvo il nuovo record come energia massima
      }
  }

  // STAMPA DI CALIBRAZIONE: Guarda questo numero sul Serial Monitor, per capire il Volume dell' ambiente circostante, serve a calibrare la max_energy
  // In teoria basterebbe ascoltare il silenzio,annotarsi il valore printato dopo "Volume Stanza: "; fare lo stesso parlando ed ascoltando
  // Infine in teoria basterebbe fare la media del volume del silenzio e del volume del tuo parlato; ma nella pratica io ho filtrato quasi al livello del parlato-
  // PER FARE IL TEST , BASTA SCOMMENTARE LE DUE RIGHE SUCCESSIVE E FARE LE OPERAZIONI DETTE ALLE RIGHE SOPRA.
  //Serial.print("Volume Stanza: ");
  //Serial.println(max_energy);
  
  // Se l'energia massima registrata in questo secondo è inferiore alla nostra 'barriera'
  if (max_energy < 220) { 
      return; // C'è solo fruscio o ci sono falsi positivi dati dai vari suoni ambientali.
              // Blocco il loop qui, così evito output inutili sul serial monitor e la scheda fa anche meno calcoli. Ovviamente l' altezza della
              // barriera è regolabile, per un filtro più potente e preciso basta alzare il valore attuale di 220(che comunque è già molto alto)
  }
  // =========================================================

  // =========================================================
  // ESECUZIONE DELLA RETE NEURALE
  // =========================================================
  // Do l'ordine all'interprete di fare la predizione ("Invoke") sui dati appena filtrati
  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Errore durante l'inferenza"); // Se l'hardware va in crash per qualche motivo, lo stampo
    return; // Interrompo questo giro
  }
  
  TfLiteTensor* output = interpreter->output(0); // Vado a copiare il risultato dall'imbuto di uscita della rete
  
  // =========================================================
  // CALCOLO DELLE PROBABILITA' PER OGNI CLASSE/PAROLA (però le filtro così da avere solo quelle più probabili)
  // =========================================================
  bool stampare_debug = false; // Creo un interruttore spento. Se non c'è nulla di interessante, non intaso lo schermo
  int percentuali[8]; // Preparo un array vuoto di 8 elementi per ospitare le percentuali finali delle mie classi
  
  for (int i = 0; i < 8; i++) { // Scorro tutte le 8 classi di uscita previste
    int8_t raw_score = output->data.int8[i]; // Prendo il punteggio grezzo della rete (formato int8, da -128 a 127)
    percentuali[i] = ((raw_score + 128) * 100) / 255; // Lo sposto in positivo (+128) e lo trasformo in una scala da 0 a 100
    
    // Se la classe attuale è una VERA parola (indice 2 o superiore) ed è passata sopra il 60% di sicurezza
    if (i >= 2 && percentuali[i] > 60) {
        stampare_debug = true; // Allora alzo l'interruttore: vale la pena stampare il report a schermo
    }
  }

  // Se l'interruttore è acceso, procedo con la stampa del report
  if (stampare_debug) {
      MicroPrintf("\n--- Analisi Modello ---"); // Titolo del report
      for (int i = 2; i < 8; i++) { // Scorro solo le VERE parole (saltando rumore e unknown)
          if (percentuali[i] >= 20) { // Filtro visivo: nascondo le parole che hanno preso meno del 20% per non riempire lo schermo
              MicroPrintf("%s: %d%%", labels[i], percentuali[i]); // Stampo il nome della parola e la sua percentuale calcolata
          }
      }
      MicroPrintf("-----------------------"); // Chiudo il report visivo
  }

  // =========================================================
  // POST-PROCESSING E CONFERMA FINALE
  // =========================================================
  const char* found_command = nullptr; // Preparo un puntatore di testo che ospiterà la parola vincente
  uint8_t score = 0; // Variabile per salvare il punteggio finale di sicurezza (0-255)
  bool is_new_command = false; // Variabile booleana che mi dirà se questa parola è "fresca" o è un rimbalzo di quella di prima
  
  // Passo l'array di probabilità grezze al modulo RecognizeCommands, che controlla lo storico e fa la media
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
      
  // Se la logica di media e smorzamento fallisce internamente
  if (process_status != kTfLiteOk) {
    MicroPrintf("Errore nell'elaborazione dei comandi"); // Avviso dell'errore
    return; // Salto il giro
  }
  
  // CONTROLLO DEFINITIVO: Se è una parola nuova AND non è la parola 'background_noises' AND non è la parola 'Unknown'
  if (is_new_command && strcmp(found_command, "background_noises") != 0 && strcmp(found_command, "Unknown") != 0) {
      // Mando a schermo in modo plateale il comando riconosciuto e confermato
      MicroPrintf(" COMANDO: %s (Sicurezza: %d/255)", found_command, score);
  }
}