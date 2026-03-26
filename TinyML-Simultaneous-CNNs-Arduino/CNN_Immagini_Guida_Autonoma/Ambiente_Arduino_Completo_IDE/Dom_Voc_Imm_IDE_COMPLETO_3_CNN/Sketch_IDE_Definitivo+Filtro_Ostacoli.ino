// =====================================================================
// BLOCCO 1: LIBRERIE E IMPORTAZIONI
// =====================================================================
#include <TensorFlowLite.h>           // Il nucleo centrale di TensorFlow Lite for Microcontrollers
#include <Arduino_OV767X.h>           // Driver ufficiale per la fotocamera TinyML integrata nella scheda
#include <PDM.h>                      // Driver per il microfono digitale (Pulse Density Modulation)
#include <new>                        // Libreria standard C++ fondamentale per usare il "Placement New" (allocazione dinamica manuale)
#include "modello_vocale_C.h"         // Il cervello addestrato per l'audio, convertito in un array C++ di numeri esadecimali
#include "modello_immagini_C.h"       // Il cervello addestrato per la vista, convertito in un array C++ di numeri esadecimali
#include "audio_provider.h"           // Modulo di Google per far comunicare il microfono con la rete neurale
#include "feature_provider.h"         // Modulo che converte l'audio raw in Spettrogrammi (Impronte vocali)
#include "micro_features_micro_model_settings.h" // Le costanti rigide: frequenze, grandezze finestre, ecc.
#include "recognize_commands.h"       // Algoritmo che filtra il rumore e decide se una parola è stata detta davvero
#include "tensorflow/lite/micro/all_ops_resolver.h" // Il dizionario che insegna alla scheda tutte le operazioni matematiche (Conv2D, Pooling, ecc.)
#include "tensorflow/lite/micro/micro_interpreter.h" // Il gestore che esegue effettivamente la rete neurale
#include "tensorflow/lite/micro/micro_log.h" // Permette di usare MicroPrintf per stampare log sul monitor seriale
#include "tensorflow/lite/micro/system_setup.h" // Funzioni di configurazione hardware a basso livello specifiche per la scheda
#include "tensorflow/lite/schema/schema_generated.h" // Definisce la struttura interna dei file di modello .tflite

// =====================================================================
// BLOCCO 2: VARIABILI GLOBALI E MEMORIA (TENSOR ARENA SHARING)
// =====================================================================
// Creo enum per far capire alla scheda in che modalità si trova
enum SystemMode {
  MODE_DOMOTICA = 0, // Stato 0: Sto ascoltando la voce
  MODE_VISIONE = 1   // Stato 1: Sto guardando con la fotocamera
};
SystemMode current_mode = MODE_DOMOTICA; // All'accensione, parto in modalità Ascolto (Domotica)

// --- ALLOCAZIONE DELLA MEMORIA CONDIVISA ---
// Allochiamo un'unica enorme scatola di memoria (120 KB)
constexpr int kTensorArenaSizeCondivisa = 120 * 1024;
// alignas(16) forza la memoria ad allinearsi a blocchi di 16 byte per far scorrere i calcoli molto più veloci sulla CPU ARM
alignas(16) uint8_t tensor_arena_condivisa[kTensorArenaSizeCondivisa]; 

// --- BUFFER FISSO PER L'INTERPRETE (uso di Placement New) ---
// Invece di usare "new" o "malloc" (che frammentano la RAM fino a far crashare Arduino),
// prepariamo un buco fisso della dimensione esatta dell'interprete. Lo svuoteremo e riempiremo a piacimento.
alignas(16) uint8_t interpreter_buffer[sizeof(tflite::MicroInterpreter)];

// Buffer Hardware: Array crudi dove mettere dentro i dati dai sensori
byte camera_buffer[176 * 144]; // La foto cruda appena scattata (Risoluzione QCIF)
int8_t shared_feature_buffer[kFeatureElementCount]; // Lo spettrogramma audio appena calcolato

// --- PUNTATORI AI MODELLI E INTERPRETI ---
// Dichiaro le variabili che ospiteranno le reti neurali, inizialmente vuote (nullptr)
const tflite::Model* model_domotica = nullptr;
const tflite::Model* model_immagini = nullptr;

tflite::MicroInterpreter* interpreter = nullptr; // L'interprete che farà girare i calcoli

TfLiteTensor* input_tensor = nullptr;  // L' ingresso dove metto foto/audio
TfLiteTensor* output_tensor = nullptr; // L' uscita dove leggo le percentuali di predizione

tflite::AllOpsResolver resolver; // Carico il dizionario delle operazioni matematiche
FeatureProvider* feature_provider_domotica = nullptr; // Oggetto per estrarre lo spettrogramma
RecognizeCommands* recognizer_domotica = nullptr; // Oggetto per capire se ho detto una parola o è solo rumore

int32_t previous_time = 0; // Tiene traccia di quando ho catturato l'ultimo audio
float g_input_scale = 0.0f; // Fattore di scala per la quantizzazione (passaggio da float a int8)
int g_input_zero_point = 0; // Punto zero per la quantizzazione (il "silenzio" in matematica intera)

// Definisco l'ordine esatto delle classi in uscita dalle due reti neurali. Devono combaciare con Python.
const char* labels_domotica[] = {"background_noises", "Unknown", "clima", "guida", "luce", "off", "on", "televisione"};
const char* labels_immagini[] = {"libero", "ostacolo"}; 

// =====================================================================
// FUNZIONE DI TRANSIZIONE (HARDWARE & MEMORY SWAP BLINDATO)
// =====================================================================
// Questa funzione cambia fisicamente il cervello caricato nella RAM.
void SwitchToMode(SystemMode mode) {
  if (current_mode == mode) return; // Se mi chiedi di andare nella modalità in cui sono già, non faccio nulla.

  // Distruggiamo l'interprete per azzerare i puntatori alla memoria
  // Chiamiamo manualmente il MicroInterpreter per liberare la Tensor Arena condivisa.
  if (interpreter != nullptr) {
    interpreter->~MicroInterpreter();
  }

  // CASO A: VOGLIO PASSARE ALLA DOMOTICA (AUDIO)
  if (mode == MODE_DOMOTICA) {
    MicroPrintf("\n=================================================");
    MicroPrintf(" RITORNO A DOMOTICA (Ignoro la Fotocamera)...");
    
    // 1. Accendo l' Hardware del Microfono. (Non spengo la fotocamera con Camera.end(), perché la libreria OV767X di Arduino ha un bug che fa crashare il bus I2C, se la spegni e riaccendi).
    PDM.begin(1, 16000); // 1 canale, 16000 Hertz
    
    // 2. PLACEMENT NEW: Instanzio la Domotica nello spazio di memoria pre-allocato
    // Scriviamo l'interprete Audio sopra al all'interprete Video nel buffer fisso. Zero frammentazione RAM.
    interpreter = new(interpreter_buffer) tflite::MicroInterpreter(model_domotica, resolver, tensor_arena_condivisa, kTensorArenaSizeCondivisa);
    
    // Chiedo all'interprete di spalmare la rete neurale nella Tensor Arena. Se la RAM non basta, si ferma.
    if (interpreter->AllocateTensors() != kTfLiteOk) { MicroPrintf("Errore Swap Domotica!"); return; }
    
    // Aggancio ingresso e uscita all'interprete appena creato 
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    // 3. Ripristino i parametri matematici necessari alla rete audio per digerire i suoni
    g_input_scale = input_tensor->params.scale;
    g_input_zero_point = input_tensor->params.zero_point;

    current_mode = MODE_DOMOTICA; // Aggiorno lo stato di sistema
    // Accendo il LED Verde sulla scheda per far capire fisicamente in che stato mi trovo. I LED sono in logica inversa (LOW = acceso)
    digitalWrite(LEDR, HIGH); digitalWrite(LEDB, HIGH); digitalWrite(LEDG, LOW); 
    previous_time = LatestAudioTimestamp(); // Reset dell'orologio interno dell'ascolto
    
    MicroPrintf(" RITORNO A DOMOTICA COMPLETATO. IN ASCOLTO...");
    MicroPrintf("=================================================");
  } 
  
  // CASO B: VOGLIO PASSARE ALLA VISIONE TRAMITE FOTOCAMERA
  else if (mode == MODE_VISIONE) {
    MicroPrintf("\n=================================================");
    MicroPrintf(" SPEGNIMENTO MICROFONO E CARICAMENTO IMMAGINI...");
    
    // 1. Spegni SOLO il Microfono PDM per liberare preziosi cicli di CPU (il microfono genererebbe migliaia di interrupt al secondo, rallentando la telecamera)
    PDM.end();
    
    // 2. PLACEMENT NEW: Instanzio le Immagini nello stesso spazio di memoria (sovrascrivo l'audio)
    interpreter = new(interpreter_buffer) tflite::MicroInterpreter(model_immagini, resolver, tensor_arena_condivisa, kTensorArenaSizeCondivisa);
    
    // Tento l'allocazione in memoria della rete video
    if (interpreter->AllocateTensors() != kTfLiteOk) { MicroPrintf("Errore Swap Immagini!"); return; }
    
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    current_mode = MODE_VISIONE; // Aggiorno lo stato
    // Accendo il LED Blu (LEDBlue).
    digitalWrite(LEDG, HIGH); digitalWrite(LEDR, HIGH); digitalWrite(LEDB, LOW); 
    
    MicroPrintf(" MODALITA' VISIONE ATTIVATA CON SUCCESSO.");
    MicroPrintf(" [Digita 's' nel Serial Monitor per fermare la telecamera]");
    MicroPrintf("=================================================");
  }
}

// =====================================================================
// BLOCCO 3: SETUP (AVVIO DELLA SCHEDA)
// =====================================================================
void setup() {
  Serial.begin(115200); // Apro il canale di comunicazione col PC ad alta velocità(i 115200 baud)
  while (!Serial); // Aspetto che il monitor seriale del PC sia effettivamente aperto prima di partire

  // Configuro i pin fisici dei tre LED a bordo scheda come Uscite di corrente
  pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, HIGH); digitalWrite(LEDG, LOW); digitalWrite(LEDB, HIGH); // Forzo il verde acceso subito

  tflite::InitializeTarget(); // Preparo i registri della CPU per la matematica di TensorFlow
  MicroPrintf("Avvio inizializzazione TFLM DUAL MODE...");

  // Inizializzo tutti e due  i sensori hardware all'avvio. Evito accensioni a posteriori, che potrebbero freezare il microcontrollore.
  if (InitAudioRecording() != kTfLiteOk) { MicroPrintf("Errore hardware Microfono!"); while(1); }
  if (!Camera.begin(QCIF, GRAYSCALE, 1)) { MicroPrintf("Errore inizializzazione Fotocamera!"); while(1); }

  // Carico gli array C++  contenenti i pesi neurali all'interno della struttura Model
  model_domotica = tflite::GetModel(modello_quantizzato_tflite);
  model_immagini = tflite::GetModel(modello_immagini_tflite);

  // Faccio la prima Allocazione di default in Modalità Audio all'accensione
  interpreter = new(interpreter_buffer) tflite::MicroInterpreter(model_domotica, resolver, tensor_arena_condivisa, kTensorArenaSizeCondivisa);
  
  // Se la Tensor Arena iniziale scoppia, blocco tutto (while(1) è un loop morto)
  if (interpreter->AllocateTensors() != kTfLiteOk) { MicroPrintf("Errore Allocazione Iniziale!"); while(1); }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  g_input_scale = input_tensor->params.scale;
  g_input_zero_point = input_tensor->params.zero_point;

  // Inizializzo gli strumenti per generare spettrogrammi (le impronte audio)
  static FeatureProvider static_fp_dom(kFeatureElementCount, shared_feature_buffer);
  feature_provider_domotica = &static_fp_dom;

  // Inizializzo il filtro anti-rumore e lo storico dei comandi detti in precedenza
  static RecognizeCommands static_recognizer_dom(1000, 140, 1000, 2);
  recognizer_domotica = &static_recognizer_dom;

  previous_time = LatestAudioTimestamp(); // Faccio partire l'ascolto
  MicroPrintf("Sistema pronto. Modalita' iniziale: DOMOTICA");
}

// =====================================================================
// BLOCCO 4: LOOP INFINITO E MACCHINA A STATI
// =====================================================================
void loop() {
  
  // --- ('s' per tornare a domotica) ---
  // Controllo se l'utente ha digitato qualcosa sulla tastiera del PC
  if (Serial.available() > 0) {
    char inChar = Serial.read(); // Leggo la lettera
    if (inChar == 's' || inChar == 'S') { // Se è la 's'...
      if (current_mode == MODE_VISIONE) { SwitchToMode(MODE_DOMOTICA); } // avvio lo switch della memoria tornando all'audio
      while(Serial.available() > 0) Serial.read(); // Svuoto la coda del cavo usb da eventuali altri caratteri spazzatura
    }
  }

  // -------------------------------------------------------------------
  // FASE 1: MODALITÀ DOMOTICA 
  // -------------------------------------------------------------------
  if (current_mode == MODE_DOMOTICA) {
    const int32_t current_time = LatestAudioTimestamp(); // Prendo il timestamp attuale
    int how_many_new_slices = 0; // Contatore di quanti millisecondi nuovi sono entrati
    
    // Richiedo a TF di aggiornare lo Spettrogramma (Impronta digitale audio) con l'audio nuovo
    TfLiteStatus feature_status = feature_provider_domotica->PopulateFeatureData(previous_time, current_time, &how_many_new_slices);
    if (feature_status != kTfLiteOk) return; // Se c'è un errore, salto questo giro di loop e riprovo
    previous_time += how_many_new_slices * kFeatureSliceStrideMs; // Sposto in avanti il cursore del tempo
    if (how_many_new_slices == 0) return; // Se non c'è audio nuovo, non faccio inferenza per non sprecare corrente

    // Copio fisicamente i pixel dello spettrogramma dentro il "Cassetto di ingresso" della rete neurale
    for (int i = 0; i < kFeatureElementCount; i++) {
      input_tensor->data.int8[i] = shared_feature_buffer[i];
    }

    // Controllo il volume totale dell'audio. Se nessuno sta parlando (silenzio totale), non accendo la rete neurale. Così risparmio anceh energia
    int max_energy = 0;
    for (int i = 0; i < kFeatureElementCount; i++) {
      int energy = abs(input_tensor->data.int8[i] - g_input_zero_point);
      if (energy > max_energy) max_energy = energy;
    }
    // Se l'energia non supera la soglia del fruscio, esco e non calcolo nulla.
    if (max_energy < 160) return;

    // Ordino all'interprete di fare i calcoli complessi ed elaborare lo spettrogramma
    if (interpreter->Invoke() != kTfLiteOk) return;

    // Variabili per raccogliere la sentenza finale della rete
    const char* found_command = nullptr;
    uint8_t score = 0;
    bool is_new_command = false; // Flag per evitare che mi ripeta "clima clima clima" se dico clima una sola volta prolungata
    
    // Passo i risultati crudi all'algoritmo intelligente che valuta se la parola è valida
    if (recognizer_domotica->ProcessLatestResults(output_tensor, current_time, &found_command, &score, &is_new_command) != kTfLiteOk) return;

    // Se ho trovato una parola vera,nuova e NON è rumore o parola da ginorare ( classe "Unknown")...
    if (found_command != nullptr && is_new_command && strcmp(found_command, "background_noises") != 0 && strcmp(found_command, "Unknown") != 0) {
      // ...Stampo a video la parola detta e il suo punteggio  x/255
      MicroPrintf(" >>> COMANDO VOCALE: %s (%d/255)", found_command, score);

      // TRIGGER: Se la parola  detta è "guida", innesco fisicamente lo Switch hardware di memoria e passo alla Fotocamera 
      if (strcmp(found_command, "guida") == 0) { SwitchToMode(MODE_VISIONE); }
    }
  }

  // -------------------------------------------------------------------
  // FASE 2: MODALITÀ VISIONE 
  // -------------------------------------------------------------------
  else if (current_mode == MODE_VISIONE) {
    
    // Scatta fisicamente la foto e la carica nel buffer  da 176x144 pixel (QCIF)
    Camera.readFrame(camera_buffer); 
    
    int input_index = 0;
    
    // Algoritmo di ritaglio live: salto le cornici esterne della foto e prendo solo un quadrato centrale perfetto di 96x96 pixel.
    // Questo è il motivo per cui l'immagine che la telecamera vede corrisponde esattamente a come abbiamo addestrato il modello in Python, addestramento scelto in base a questo
    for (int y = 24; y < 120; y++) {
      for (int x = 40; x < 136; x++) {
        // Calcolo su che pixel della foto originale sono e prendo il valore di luminosità del singolo pixel
        int buffer_index = (y * 176) + x; 
        // Conversione matematica critica: i pixel escono dalla cam tra 0 e 255 (unsigned).
        // La rete neurale Quantizzata int8 lavora tra -128 e +127 (signed). Tolgo 128 per centrare lo zero. Copio nel cassetto d'ingresso.
        input_tensor->data.int8[input_index++] = camera_buffer[buffer_index] - 128;
      }
    }

    // --- RADAR DI DEBUG ---
    // Estraggo il valore di luce del pixel esatto che sta al centro geometrico del ritaglio.
    // Debug per capire se l'ambiente è troppo buio per il sensore, o se il sensore vede tutto nero perchè non sta funzionando
    int8_t pixel_centro = input_tensor->data.int8[(96 * 96) / 2];
    
    // Faccio guardare l'immagine ritagliata alla Rete Neurale e le faccio calcolare i risultati
    if (interpreter->Invoke() != kTfLiteOk) return;

    // --- LETTURA DIRETTA DELLE 2 CLASSI ---
    // Invece di cercare il massimo, estraggo i voti esatti per entrambe le classi
    // [0] = libero, [1] = ostacolo (grazie all'ordine alfabetico)
    
    // Converto subito i valori raw (-128 a 127) in percentuali (0-100)
    int score_libero = ((output_tensor->data.int8[0] + 128) * 100) / 255;
    int score_ostacolo = ((output_tensor->data.int8[1] + 128) * 100) / 255;

    // --- SOGLIA ASIMMETRICA  ---
    // Impongo che l'ostacolo debba superare una soglia altissima per essere creduto.
    // Se la rete dice "Ostacolo al 70%", l' output non viene generato se la mia soglia è 97% ad esempio. 
    // Modificare questo "97" per rendere il modello più o meno tollerante verso gli ostacoli.
    if (score_ostacolo > 97) { 
        MicroPrintf("[VISIONE] OSTACOLO RILEVATO! (Sicurezza: %d%%) | Luce: %d", score_ostacolo, pixel_centro);
        digitalWrite(LEDB, HIGH); digitalWrite(LEDR, LOW); // STOP = LED Rosso
    } 
    else {
        // Se non c'è un ostacolo al 98+%, considero la strada libera
        MicroPrintf("[VISIONE] Strada Libera (Probabilita' ostacolo solo al %d%%) | Luce: %d", score_ostacolo, pixel_centro);
        digitalWrite(LEDR, HIGH); digitalWrite(LEDB, LOW); // LIBERO = LED Blu
    }
  }
}