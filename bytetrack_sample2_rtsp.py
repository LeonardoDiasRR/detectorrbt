import cv2
from ultralytics import YOLO

# 1. Configurações
# Substitua pelo URL da sua câmera RTSP. Use um placeholder caso não tenha.
RTSP_URL = "rtsp://findface:Mitra2021@10.95.7.24:7001/022cdcf4-b31d-f379-c2d8-b4e727f8a148"
BATCH_SIZE = 4  # Processamento em lote a cada 4 frames
CONFIDENCE_THRESHOLD = 0.3  # Índice de confiança mínimo para rastreamento
MODEL_PATH = 'yolov8n-face.pt'  # Modelo YOLO a ser carregado (ex: nano)
TRACKER_CONFIG = 'bytetrack.yaml' # Arquivo de configuração para ByteTrack

# 2. Carregar o Modelo e Inicializar o Rastreamento
print(f"Carregando modelo: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 3. Executar o Rastreamento com RTSP e Batch
# O modo 'track' da Ultralytics gerencia a leitura do stream (RTSP), 
# a detecção, o rastreamento (ByteTrack) e, crucialmente, 
# o processamento em lote (batch).
print(f"Iniciando rastreamento no stream RTSP: {RTSP_URL}")
print(f"Tamanho do Batch: {BATCH_SIZE} frames")

# O argumento 'source' aceita URLs RTSP diretamente.
# O argumento 'batch' define o tamanho do lote para inferência.
# O argumento 'stream=True' garante que o processamento seja otimizado para streams/vídeos.
# O argumento 'tracker' especifica o arquivo de configuração do tracker (ByteTrack é o padrão).
# O argumento 'show=True' exibe automaticamente os resultados anotados em tempo real.

try:
    results = model.track(
        source=RTSP_URL,
        conf=CONFIDENCE_THRESHOLD,
        tracker=TRACKER_CONFIG,
        batch=BATCH_SIZE,  # Define o processamento em lote de frames
        show=True,         # Exibe os frames anotados
        stream=True        # Otimiza para stream de vídeo (RTSP)
    )

    # Nota: Com 'show=True', o loop `for result in results:` é opcional para exibição, 
    # mas seria usado para processamento posterior dos resultados (ex: salvar em BD).
    # Como queremos apenas mostrar na tela, o 'show=True' faz o trabalho principal.
    
    # Exemplo de como acessar os resultados (executado se 'stream=True' retornar um gerador):
    # for r in results:
    #     boxes = r.boxes.xyxy.cpu().numpy() # Bounding boxes
    #     track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [] # IDs de rastreamento
    #     confidences = r.boxes.conf.cpu().numpy() # Índices de confiança
    #     # O método 'plot()' do objeto Result já cria o frame anotado
    #     annotated_frame = r.plot()
    #     cv2.imshow("Rastreamento YOLO + ByteTrack", annotated_frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    pass

except Exception as e:
    print(f"\n❌ Erro durante o rastreamento ou conexão com o stream RTSP: {e}")
    print("Verifique se o URL RTSP está correto e o stream está ativo.")

finally:
    # Garantir que todas as janelas do OpenCV sejam fechadas ao terminar
    cv2.destroyAllWindows()
    print("\nProcesso de rastreamento finalizado.")