// =====================================================================
// BLOCCO 1: LIBRERIE E VARIABILI DI MEMORIA
// =====================================================================
// Questo script ci permette di usare il microfono dell' arduino in combinazione al prossimo script da lanciare in ambiente Python
// Catturiamo il suono e lo mettiamo in un contenitore di RAM.

#include <PDM.h> // Importo la libreria ufficiale per gestire il microfono digitale,integrato nella scheda

// Parametri Audio Hardware
static const char channels = 1; // Forzo la registrazione a un singolo canale (Mono)
static const int frequency = 16000; // Imposto la frequenza a 16kHz (lo standard che sarà richiesto dalla nostra rete neurale  )
const int TOTAL_SAMPLES = 32000; // Registro 32.000 campioni totali (che a 16kHz corrispondono a 2 secondi interi di audio, così da cattuarare al meglio le parole registrate)

// Buffer di memoria nella RAM dell'Arduino
short audioBuffer[TOTAL_SAMPLES]; // Alloco un blocco molto grande di RAM per salvare tutta la registrazione. 'short' significa numeri interi a 16-bit.
short sampleBuffer[512]; // Un piccolo buffer temporaneo che fa da "imbuto" per raccogliere i dati  dal microfono

// Variabili di stato (Flag). 
// Usiamo 'volatile' perché queste variabili vengono modificate dall'hardware in background (interrupt) e non dal normale flusso di codice.
volatile int bufferIndex = 0; // Indice/cursore per sapere a che punto di riempimento del buffer grande siamo arrivati
volatile bool recording = false; // Interruttore per accendere/spegnere materialmente il salvataggio dei dati
volatile bool readyToSend = false; // Bandierina che si alza quando il buffer è pieno e l'audio è pronto per essere inviato al PC

// =====================================================================
// BLOCCO 2: SETUP INIZIALE
// =====================================================================
void setup() {
  Serial.begin(115200); // Apro la comunicazione USB con il computer ad altissima velocità per non fare da blocco quando invio i dati
  while (!Serial); // Blocco la scheda: non fa niente finché non apro il Monitor Seriale o avvio lo script Python sul PC

  // Configura il microfono integrato
  PDM.onReceive(onPDMdata); // Aggancio il microfono alla mia funzione in basso: ogni volta che sente qualcosa, la fa scattare in automatico
  if (!PDM.begin(channels, frequency)) { // Tento di accendere fisicamente il microfono con le mie regole (Mono, 16kHz)
    Serial.println("Errore nell'inizializzazione del microfono PDM!"); // Se è guasto o occupato, genero un avviso 
    while (1); // E blocco la scheda in un loop infinito di sicurezza
  }
}

// =====================================================================
// BLOCCO 3: LOOP PRINCIPALE 
// =====================================================================
// Il loop non registra fisicamente l'audio, fa solo da supervisore e smistatore:
// aspetta il comando dal PC e spedisce i dati quando sono pronti.

void loop() {
  // 1. In attesa del comando 'R' dal PC, che all' effettivo significa spingere invio sulla tastiera quando eseguo lo script Python
  if (Serial.available() > 0) { // Controllo se il cavo USB ha portato qualche messaggio dal computer
    char c = Serial.read(); // Leggo l'esatto carattere inviato dal PC
    
    // Se il PC mi ha mandato una 'R' (Record) E non sto già registrando E non sto già inviando dati...
    if (c == 'R' && !recording && !readyToSend) { 
      bufferIndex = 0; // Riporto il cursore a zero (così sovrascrivo l'audio vecchio)
      recording = true; // Premo il tasto "REC". Da ora la funzione hardware inizierà a salvare
    }
  }

  // 2. Registrazione finita, invia i dati binari al PC alla massima velocità
  if (readyToSend) { // Se la funzione hardware in basso ha alzato questa bandierina...
    // Mando al PC l'intero blocco di RAM; 
    // Moltiplico TOTAL_SAMPLES * 2 perché ogni campione 'short' occupa fisicamente 2 byte.
    Serial.write((uint8_t*)audioBuffer, TOTAL_SAMPLES * 2); 
    readyToSend = false; // Invio finito: abbasso la bandierina e mi rimetto in attesa di un'altra 'R' dal PC
  }
}

// =====================================================================
// BLOCCO 4: ACQUISIZIONE HARDWARE (INTERRUPT)
// =====================================================================
// Questa funzione non viene mai chiamata nel loop. Si attiva autonomamente in background
// ogni volta che il microfono ha un nuovo pacchetto di suono pronto.

void onPDMdata() {
  int bytesAvailable = PDM.available(); // Chiede al microfono i byte del nuovo suono che ha accumulato
  PDM.read(sampleBuffer, bytesAvailable); // sposto questi byte grezzi dal microfono al mio piccolo imbuto (sampleBuffer)
  
  // Divido i byte per 2 per calcolare quanti campioni audio sono effettivamente entrati (visto che pesano 2 byte l'uno)
  int samples = bytesAvailable / 2; 

  if (recording) { // Se il tasto REC è premuto (grazie al comando 'R' dal PC)...
    for (int i = 0; i < samples; i++) { // Scorro uno ad uno i campioni appena entrati nell'imbuto
      if (bufferIndex < TOTAL_SAMPLES) { // Controllo se ho ancora spazio nel mio grande contenitore di RAM
        audioBuffer[bufferIndex++] = sampleBuffer[i]; // se si, sposto il campione nel contenitore e muovo il cursore in avanti di un passo
      }
    }
    
    // >If per verificare se ho raggiunto il limite di registrazione ( 32000 spazi)
    if (bufferIndex >= TOTAL_SAMPLES) {
      recording = false; // Spengo automaticamente il REC
      readyToSend = true; // Alzo la bandierina per dire al loop principale di inviare il pacchetto pronto al PC 
    }
  }
}