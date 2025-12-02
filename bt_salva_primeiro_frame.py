import cv2
import os
from ultralytics import YOLO

# --- 1. Configurações ---
RTSP_URL = "rtsp://findface:Mitra2021@10.95.7.24:7001/022cdcf4-b31d-f379-c2d8-b4e727f8a148"
OUTPUT_DIR = './imagens'
MODEL_PATH = 'yolov8n-face.pt'  
TRACKER_CONFIG = 'bytetrack.yaml' 
CONFIDENCE_THRESHOLD = 0.3

# Garante que o diretório de saída exista
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Diretório de saída '{OUTPUT_DIR}' garantido.")

# Conjunto (Set) para armazenar os IDs já rastreados e salvos
tracked_ids_saved = set()

# --- 2. Carregar o Modelo ---
model = YOLO(MODEL_PATH)

# --- 3. Função de Processamento do Stream ---
print(f"Iniciando rastreamento no stream RTSP...")

# O stream=True faz com que model.track() retorne um gerador, 
# permitindo o processamento frame a frame dentro do loop.
results_generator = model.track(
    source=RTSP_URL,
    conf=CONFIDENCE_THRESHOLD,
    tracker=TRACKER_CONFIG,
    stream=True,  # Essencial para processamento frame a frame com lógica customizada
    show=False    # Desativamos o show=True nativo para gerenciar o loop de exibição
)

try:
    for result in results_generator:
        # Obter o frame anotado (com bboxes, IDs e conf)
        annotated_frame = result.plot()
        
        # --- Lógica de Salvamento do Primeiro Frame ---
        if result.boxes.id is not None:
            # Obtém os IDs de rastreamento (ByteTrack)
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            
            for track_id in track_ids:
                # Se o ID for novo (não está no conjunto)
                if track_id not in tracked_ids_saved:
                    # Monta o nome do arquivo
                    filename = os.path.join(OUTPUT_DIR, f"track-{track_id}.jpg")
                    
                    # Salva o frame atual (frame completo anotado)
                    cv2.imwrite(filename, annotated_frame)
                    
                    # Adiciona o ID ao conjunto para não salvar novamente
                    tracked_ids_saved.add(track_id)
                    print(f"Novo objeto detectado! Primeiro frame salvo como: {filename}")
        
        # --- Exibição na Tela ---
        cv2.imshow("Rastreamento YOLO + ByteTrack", annotated_frame)
        
        # Parar se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"\n❌ Erro durante o rastreamento ou conexão: {e}")
    print("Verifique o URL RTSP e o status do stream.")

finally:
    cv2.destroyAllWindows()
    print("\nProcesso de rastreamento finalizado.")
    print(f"Total de objetos rastreados (com primeiro frame salvo): {len(tracked_ids_saved)}")