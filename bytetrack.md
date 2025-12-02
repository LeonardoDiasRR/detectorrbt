# Configurações do Rastreamento de Múltiplos Objetos (MOT) - ByteTrack

# 1. Parâmetros de Detecção e Limites
# ------------------------------------

# Limite de confiança para detecções que são consideradas "High Score" (Alta Pontuação).
# Essas detecções são usadas para inicializar e atualizar trilhas primárias.
track_high_thresh: 0.3

# Limite de confiança para detecções que são consideradas "Low Score" (Baixa Pontuação).
# Detecções entre track_low_thresh e track_high_thresh são usadas para 
# recuperar trilhas que foram temporariamente perdidas (second stage association).
track_low_thresh: 0.1

# Limite IOU (Intersection Over Union) para a associação de detecções.
# Se o IOU entre uma trilha e uma detecção for menor que este valor,
# a detecção não será associada àquela trilha.
iou_thresh: 0.7


# 2. Parâmetros de Gerenciamento de Trilha (Track Management)
# -----------------------------------------------------------

# Limite de confiança para inicializar uma nova trilha.
# Uma detecção deve ter confiança superior a este valor para iniciar um novo ID.
new_track_thresh: 0.1

# Número de frames que uma trilha pode ficar sem ser atualizada 
# (não associada a uma nova detecção) antes de ser considerada "morta" e removida.
track_buffer: 30

# Mínimo de frames em que uma trilha deve existir para ser considerada "confirmada" 
# e incluída nos resultados finais.
min_box_area: 10 # Tamanho mínimo (em pixels²) da bounding box para ser rastreada.


# 3. Outros Parâmetros (Gerais)
# -----------------------------

# Método de similaridade/distância usado para associar caixas.
# 'iou' é o padrão. Para ByteTrack, geralmente é baseado em IOU.
match_thresh: 0.9

# Tipo de modelo de movimento usado pelo filtro de Kalman (se aplicável).
# Em ByteTrack, o filtro de Kalman é usado para prever a posição futura da trilha.
# Padrão para 'bytetrack.yaml' é frequentemente um modelo de movimento constante.
kalman_sigma_a: 0.1 # Desvio padrão para aceleração (ruído do modelo).
kalman_sigma_q: 0.05 # Desvio padrão para o ruído de observação.