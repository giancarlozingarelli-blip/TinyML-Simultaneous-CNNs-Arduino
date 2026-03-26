import cv2
import os
import yt_dlp
import glob

# =====================================================================
# IMPOSTAZIONI DEL PROGETTO
# =====================================================================
URL_YOUTUBE = "INSERIRE QUI IL LINK DEL VIDEO"  # Esempio: "https://www.youtube.com/watch?v=xyU7qrJ-lpQ"

CARTELLA_DESTINAZIONE = "dati_libero"
NOME_VIDEO_TEMP = "video_temporaneo.mp4"

# ---------------------------------------------------------------------
# IMPOSTAZIONI DI TEMPO (Da dove a dove estrarre)
# ---------------------------------------------------------------------
MINUTO_INIZIO = 12
SECONDO_INIZIO = 0  # Inizio in  minuti e secondi

MINUTO_FINE = 16
SECONDO_FINE = 0     # Stop in minuti e  secondi

# Foto massime che si vuole estrarre nell intervallo scelto 
MAX_IMMAGINI = 2000
# Ogni quanti secondi viene scattata la foto
ESTRAI_OGNI_SECONDI = 1  

# =====================================================================
# MOTORE DI ESTRAZIONE
# =====================================================================
def scarica_ed_estrai():
    # 1. Creiamo la cartella se non esiste
    os.makedirs(CARTELLA_DESTINAZIONE, exist_ok=True)

    # 2. CALCOLO DELL'INDICE DI PARTENZA (Per non sovrascrivere foto vecchie)
    file_esistenti = glob.glob(os.path.join(CARTELLA_DESTINAZIONE, "*.jpg"))
    indice_partenza = len(file_esistenti)
    
    print(f"[*] Trovate {indice_partenza} immagini già esistenti.")
    print(f"[*] Le nuove immagini ripartiranno dal numero {indice_partenza}.")

    # 3. Scarichiamo il video da YouTube (FORZANDO IL CODEC H.264)
    ydl_opts = {
        'format': 'bestvideo[vcodec^=avc1][height<=480][ext=mp4]/best[vcodec^=avc1]', 
        'outtmpl': NOME_VIDEO_TEMP,
        'quiet': False
    }

    print("\n[*] Fase 1: Download del video da YouTube in corso...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([URL_YOUTUBE])
    except Exception as e:
        print(f"Errore durante il download: {e}")
        return

    print(f"\n[*] Fase 2: Estrazione dal minuto {MINUTO_INIZIO}:{SECONDO_INIZIO:02d} al {MINUTO_FINE}:{SECONDO_FINE:02d}...")
    
    # Apriamo il video scaricato con OpenCV
    video = cv2.VideoCapture(NOME_VIDEO_TEMP)
    
    # Capiamo a quanti FPS è il video per fare i calcoli di tempo
    fps = round(video.get(cv2.CAP_PROP_FPS))
    if fps == 0: 
        fps = 30
        
    salta_frame = fps * ESTRAI_OGNI_SECONDI
    
    # Traduciamo i minuti/secondi in "Numero di Frame"
    inizio_sec = (MINUTO_INIZIO * 60) + SECONDO_INIZIO
    fine_sec = (MINUTO_FINE * 60) + SECONDO_FINE
    
    frame_iniziale = inizio_sec * fps
    frame_finale = fine_sec * fps
    
    # Spostiamo il video direttamente al frame di inizio 
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_iniziale)
    
    conteggio_frame = frame_iniziale
    immagini_estratte_ora = 0
    
    while True:
        successo, frame = video.read()
        
        # Ci fermiamo se: il video finisce, superiamo il minuto di fine, o raggiungiamo il limite foto
        if not successo or conteggio_frame > frame_finale or immagini_estratte_ora >= MAX_IMMAGINI:
            break
            
        # Salviamo la foto solo se è il momento giusto
        # Usiamo (conteggio_frame - frame_iniziale) così prende subito la primissima foto utile
        if (conteggio_frame - frame_iniziale) % salta_frame == 0:
            
            numero_foto = indice_partenza + immagini_estratte_ora
            nome_foto = os.path.join(CARTELLA_DESTINAZIONE, f"vialibera_{numero_foto:05d}.jpg")
            
            cv2.imwrite(nome_foto, frame)
            
            immagini_estratte_ora += 1
            if immagini_estratte_ora % 100 == 0:
                # Stampo anche il minuto a cui è arrivato per darti un riferimento visivo
                minuto_corrente = conteggio_frame // (fps * 60)
                secondo_corrente = (conteggio_frame // fps) % 60
                print(f" -> Estratte {immagini_estratte_ora} foto (Ora in elaborazione: {minuto_corrente}:{secondo_corrente:02d})...")
                
        conteggio_frame += 1

    # Rilascio del video
    video.release()
    
    # 4. cancelliamo il video dal computer
    if os.path.exists(NOME_VIDEO_TEMP):
        os.remove(NOME_VIDEO_TEMP)
        
    print("\n=======================================================")
    print(f" SUCCESSO! Hai estratto {immagini_estratte_ora} nuove foto dall'intervallo richiesto.")
    print(f" Ora la tua cartella '{CARTELLA_DESTINAZIONE}' contiene un TOTALE di {indice_partenza + immagini_estratte_ora} immagini.")
    print("=======================================================")

if __name__ == "__main__":
    if URL_YOUTUBE == "INSERIRE QUI IL LINK DEL VIDEO":
        print("ERRORE: Incollare un link di YouTube alla riga 9 dello script!")
    else:
        scarica_ed_estrai()