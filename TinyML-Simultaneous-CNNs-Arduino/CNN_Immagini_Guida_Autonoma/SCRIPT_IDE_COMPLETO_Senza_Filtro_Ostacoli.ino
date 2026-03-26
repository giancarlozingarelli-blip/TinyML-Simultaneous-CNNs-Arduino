// =====================================================================
// BLOCCO 1: LIBRERIE E IMPORTAZIONI
// =====================================================================
#include <TensorFlowLite.h>
#include <Arduino_OV767X.h>                   
#include <PDM.h>                              
#include <new>                                // Necessario per il Placement New
#include "modello_vocale_C.h"                 
#include "modello_immagini_C.h"               
#include "audio_provider.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// =====================================================================
// BLOCCO 2: VARIABILI GLOBALI E MEMORIA (TENSOR ARENA SHARING)
// =====================================================================
enum SystemMode {
  MODE_DOMOTICA = 0,
  MODE_VISIONE = 1
};
SystemMode current_mode = MODE_DOMOTICA; 

// --- ALLOCAZIONE DELLA MEMORIA CONDIVISA ---
constexpr int kTensorArenaSizeCondivisa = 120 * 1024;
alignas(16) uint8_t tensor_arena_condivisa[kTensorArenaSizeCondivisa];

// --- BUFFER FISSO PER L'INTERPRETE (Il trucco del Placement New) ---
alignas(16) uint8_t interpreter_buffer[sizeof(tflite::MicroInterpreter)];

// Buffer Hardware
byte camera_buffer[176 * 144];
int8_t shared_feature_buffer[kFeatureElementCount];

// --- PUNTATORI AI MODELLI E INTERPRETI ---
const tflite::Model* model_domotica = nullptr;
const tflite::Model* model_immagini = nullptr;

tflite::MicroInterpreter* interpreter = nullptr; 

TfLiteTensor* input_tensor = nullptr;  
TfLiteTensor* output_tensor = nullptr;

tflite::AllOpsResolver resolver;
FeatureProvider* feature_provider_domotica = nullptr;
RecognizeCommands* recognizer_domotica = nullptr;

int32_t previous_time = 0;
float g_input_scale = 0.0f;
int g_input_zero_point = 0;

const char* labels_domotica[] = {"background_noises", "Unknown", "clima", "guida", "luce", "off", "on", "televisione"};
const char* labels_immagini[] = {"libero", "ostacolo"}; 

// =====================================================================
// FUNZIONE DI TRANSIZIONE (HARDWARE & MEMORY SWAP BLINDATO)
// =====================================================================
void SwitchToMode(SystemMode mode) {
  if (current_mode == mode) return;

  // Distruggiamo l'interprete per azzerare i puntatori alla memoria
  if (interpreter != nullptr) {
    interpreter->~MicroInterpreter();
  }

  if (mode == MODE_DOMOTICA) {
    MicroPrintf("\n=================================================");
    MicroPrintf(" RITORNO A DOMOTICA (Ignoro la Fotocamera)...");
    
    // 1. Accendi Hardware del Microfono (La fotocamera NON viene spenta con end() per evitare crash I2C)
    PDM.begin(1, 16000);
    
    // 2. PLACEMENT NEW: Instanzio la Domotica nello spazio di memoria pre-allocato
    interpreter = new(interpreter_buffer) tflite::MicroInterpreter(model_domotica, resolver, tensor_arena_condivisa, kTensorArenaSizeCondivisa);
    
    if (interpreter->AllocateTensors() != kTfLiteOk) { MicroPrintf("Errore Swap Domotica!"); return; }
    
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    // 3. Ripristina parametri audio
    g_input_scale = input_tensor->params.scale;
    g_input_zero_point = input_tensor->params.zero_point;

    current_mode = MODE_DOMOTICA;
    digitalWrite(LEDR, HIGH); digitalWrite(LEDB, HIGH); digitalWrite(LEDG, LOW); // LED Verde
    previous_time = LatestAudioTimestamp();
    
    MicroPrintf(" RITORNO A DOMOTICA COMPLETATO. IN ASCOLTO...");
    MicroPrintf("=================================================");
  } 
  else if (mode == MODE_VISIONE) {
    MicroPrintf("\n=================================================");
    MicroPrintf(" SPEGNIMENTO MICROFONO E CARICAMENTO IMMAGINI...");
    
    // 1. Spegni SOLO il Microfono PDM per liberare la CPU
    PDM.end();
    
    // 2. PLACEMENT NEW: Instanzio le Immagini nello stesso spazio di memoria
    interpreter = new(interpreter_buffer) tflite::MicroInterpreter(model_immagini, resolver, tensor_arena_condivisa, kTensorArenaSizeCondivisa);
    
    if (interpreter->AllocateTensors() != kTfLiteOk) { MicroPrintf("Errore Swap Immagini!"); return; }
    
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    current_mode = MODE_VISIONE;
    digitalWrite(LEDG, HIGH); digitalWrite(LEDR, HIGH); digitalWrite(LEDB, LOW); // LED Blu
    
    MicroPrintf(" MODALITA' VISIONE ATTIVATA CON SUCCESSO.");
    MicroPrintf(" [Digita 's' nel Serial Monitor per fermare la telecamera]");
    MicroPrintf("=================================================");
  }
}

// =====================================================================
// BLOCCO 3: SETUP (AVVIO DELLA SCHEDA)
// =====================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, HIGH); digitalWrite(LEDG, LOW); digitalWrite(LEDB, HIGH); // Verde

  tflite::InitializeTarget();
  MicroPrintf("Avvio inizializzazione TFLM DUAL MODE...");

  // Inizializzo ENTRAMBI i sensori all'avvio per non bloccare il bus in seguito
  if (InitAudioRecording() != kTfLiteOk) { MicroPrintf("Errore hardware Microfono!"); while(1); }
  if (!Camera.begin(QCIF, GRAYSCALE, 1)) { MicroPrintf("Errore inizializzazione Fotocamera!"); while(1); }

  model_domotica = tflite::GetModel(modello_quantizzato_tflite);
  model_immagini = tflite::GetModel(modello_immagini_tflite);

  // Allocazione Iniziale in Modalità Domotica
  interpreter = new(interpreter_buffer) tflite::MicroInterpreter(model_domotica, resolver, tensor_arena_condivisa, kTensorArenaSizeCondivisa);
  
  if (interpreter->AllocateTensors() != kTfLiteOk) { MicroPrintf("Errore Allocazione Iniziale!"); while(1); }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  g_input_scale = input_tensor->params.scale;
  g_input_zero_point = input_tensor->params.zero_point;

  static FeatureProvider static_fp_dom(kFeatureElementCount, shared_feature_buffer);
  feature_provider_domotica = &static_fp_dom;

  static RecognizeCommands static_recognizer_dom(1000, 180, 1000, 2);
  recognizer_domotica = &static_recognizer_dom;

  previous_time = LatestAudioTimestamp();
  MicroPrintf("Sistema pronto. Modalita' iniziale: DOMOTICA");
}

// =====================================================================
// BLOCCO 4: LOOP INFINITO E MACCHINA A STATI
// =====================================================================
void loop() {
  
  // --- ASCOLTO SERIALE ('s' per tornare a domotica) ---
  if (Serial.available() > 0) {
    char inChar = Serial.read();
    if (inChar == 's' || inChar == 'S') {
      if (current_mode == MODE_VISIONE) { SwitchToMode(MODE_DOMOTICA); }
      while(Serial.available() > 0) Serial.read(); 
    }
  }

  // -------------------------------------------------------------------
  // FASE 1: MODALITÀ DOMOTICA
  // -------------------------------------------------------------------
  if (current_mode == MODE_DOMOTICA) {
    const int32_t current_time = LatestAudioTimestamp();
    int how_many_new_slices = 0;
    
    TfLiteStatus feature_status = feature_provider_domotica->PopulateFeatureData(previous_time, current_time, &how_many_new_slices);
    if (feature_status != kTfLiteOk) return;
    previous_time += how_many_new_slices * kFeatureSliceStrideMs;
    if (how_many_new_slices == 0) return;

    for (int i = 0; i < kFeatureElementCount; i++) {
      input_tensor->data.int8[i] = shared_feature_buffer[i];
    }

    int max_energy = 0;
    for (int i = 0; i < kFeatureElementCount; i++) {
      int energy = abs(input_tensor->data.int8[i] - g_input_zero_point);
      if (energy > max_energy) max_energy = energy;
    }
    if (max_energy < 120) return;

    if (interpreter->Invoke() != kTfLiteOk) return;

    const char* found_command = nullptr;
    uint8_t score = 0;
    bool is_new_command = false;
    
    if (recognizer_domotica->ProcessLatestResults(output_tensor, current_time, &found_command, &score, &is_new_command) != kTfLiteOk) return;

    if (found_command != nullptr && is_new_command && strcmp(found_command, "background_noises") != 0 && strcmp(found_command, "Unknown") != 0) {
      MicroPrintf(" >>> COMANDO VOCALE: %s (%d/255)", found_command, score);

      if (strcmp(found_command, "guida") == 0) { SwitchToMode(MODE_VISIONE); }
    }
  }

  // -------------------------------------------------------------------
  // FASE 2: MODALITÀ VISIONE
  // -------------------------------------------------------------------
  else if (current_mode == MODE_VISIONE) {
    
    Camera.readFrame(camera_buffer); 
    
    int input_index = 0;
    for (int y = 24; y < 120; y++) {
      for (int x = 40; x < 136; x++) {
        int buffer_index = (y * 176) + x; 
        input_tensor->data.int8[input_index++] = camera_buffer[buffer_index] - 128;
      }
    }

    // --- RADAR DI DEBUG ---
    // Estraiamo il pixel esatto al centro dell'immagine
    int8_t pixel_centro = input_tensor->data.int8[(96 * 96) / 2];
    
    if (interpreter->Invoke() != kTfLiteOk) return;

    int8_t max_score_img = -128;
    int max_index_img = 0;

    for (int i = 0; i < 2; i++) {
      if (output_tensor->data.int8[i] > max_score_img) {
        max_score_img = output_tensor->data.int8[i];
        max_index_img = i;
      }
    }
    
    int percentuale_img = ((max_score_img + 128) * 100) / 255;
    if(percentuale_img > 60) {
        MicroPrintf("[VISIONE] %s (%d%%) | Luce Centro: %d", labels_immagini[max_index_img], percentuale_img, pixel_centro);
        
        if (strcmp(labels_immagini[max_index_img], "ostacolo") == 0) {
            digitalWrite(LEDB, HIGH); digitalWrite(LEDR, LOW); // STOP = LED Rosso
        } else {
            digitalWrite(LEDR, HIGH); digitalWrite(LEDB, LOW); // LIBERO = LED Blu
        }
    }
  }
}