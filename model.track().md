# Manual de Parâmetros do Método `model.track()` — Ultralytics YOLOv8
Guia completo dos argumentos aceitos pelo método `model.track()` na versão atual do YOLOv8, organizados por funcionalidade e incluindo os parâmetros específicos do algoritmo de tracking ByteTrack.

---

## 1. Entrada e Saída

### 1.1 Fonte de Dados
- **`source`**  
  Caminho ou entrada de vídeo/imagem: arquivo `.mp4`, `.avi`, `.mkv`, câmera RTSP, HTTP, webcam, diretório de imagens etc.

- **`stream`** *(bool)*  
  Ativa retorno frame-a-frame (gerador).

---

### 1.2 Salvamento de Resultados
- **`save`** *(bool)*  
  Salva automaticamente os vídeos/imagens de saída.

- **`save_frames`** *(bool)*  
  Salva cada frame individual processado.

- **`save_txt`** *(bool)*  
  Salva caixas de detecção em arquivos TXT no formato YOLO.

- **`save_crop`** *(bool)*  
  Salva recortes (crops) dos objetos detectados/rastreados.

- **`project`**  
  Define o diretório raiz onde os resultados serão salvos.

- **`name`**  
  Subpasta dentro de `project/`.

- **`exist_ok`** *(bool)*  
  Permite sobrescrever resultados sem erro.

---

## 2. Configurações do Modelo

### 2.1 Tamanho e Dispositivo
- **`imgsz`**  
  Tamanho da imagem de inferência (ex.: 640, 1280).

- **`device`**  
  `"cpu"`, `"cuda"`, `"0"`, `"0,1"`, etc.

- **`half`** *(bool)*  
  Inferência em FP16 (somente GPUs NVIDIA).

---

### 2.2 Thresholds de Detecção
- **`conf`**  
  Limite mínimo de confiança.

- **`iou`**  
  Threshold de Interseção sobre União para NMS.

- **`max_det`**  
  Máximo de detecções por frame.

- **`classes`**  
  Filtrar por IDs de classes específicas.

- **`agnostic_nms`** *(bool)*  
  NMS sem considerar classes.

---

## 3. Configuração do Rastreamento

### 3.1 Parâmetros Principais
- **`tracker`**  
  Caminho para arquivo `.yaml` de configuração do tracker  
  *(ex.: bytetrack.yaml ou botsort.yaml)*.

- **`persist`** *(bool)*  
  Mantém IDs consistentes mesmo após interrupções ou ausência temporária de detecção.

- **`reid_model`**  
  Modelo de Re-ID (reidentificação) quando suportado pelo tracker.

---

## 4. Visualização e Renderização

- **`show`** *(bool)*  
  Exibe janelas com os resultados enquanto processa.

- **`visualize`** *(bool)*  
  Gera visualizações internas (featuremaps, etc.).

- **`line_width`**  
  Espessura das caixas e linhas de rastreamento.

- **`hide_labels`** *(bool)*  
  Oculta nomes das classes.

- **`hide_conf`** *(bool)*  
  Oculta valores de confiança.

- **`vid_stride`**  
  Processa apenas 1 a cada N frames para acelerar.

---

## 5. Performance e Debugging

- **`verbose`** *(bool)*  
  Exibe logs detalhados durante o processo.

- **`profile`** *(bool)*  
  Coleta tempos de inferência.

- **`augment`** *(bool)*  
  Ativa inferência com augmentations (custa desempenho).

- **`retina_masks`** *(bool)*  
  Máscaras de alta resolução (apenas YOLOv8-seg).

---

## 6. Argumentos Avançados

- **`bbox`**  
  Usar detecções pré-existentes em vez de rodar o detector.

- **`format`**  
  Formato de saída (JSON/CSV, quando disponível).

- **`callback`**  
  Função chamada a cada frame processado.

---

# 7. Parâmetros do ByteTrack (arquivo `bytetrack.yaml`)

O método `model.track()` aceita um arquivo `.yaml` contendo parâmetros do algoritmo ByteTrack.  
Abaixo, a lista completa dos parâmetros reconhecidos e utilizados pelo Ultralytics:

### 7.1 Parâmetros de Associação
- **`track_high_thresh`**  
  Confiança mínima para associação primária (matching de alta confiança).  
  *Default:* 0.5

- **`track_low_thresh`**  
  Confiança mínima para manter “pseudo-detecções” ou objetos considerados fracos.  
  *Default:* 0.1

- **`new_track_thresh`**  
  Confiança mínima para iniciar uma nova track.  
  *Default:* 0.6

---

### 7.2 Parâmetros de Re-Identificação (ReID)
- **`match_thresh`**  
  Threshold do custo IoU para matching definitivo entre detecções e tracks.  
  *Default:* 0.8

---

### 7.3 Parâmetros de Persistência
- **`track_buffer`**  
  Número de frames que um objeto pode ficar “desaparecido” antes de ser descartado.  
  *Default:* 30

- **`min_box_area`**  
  Área mínima da caixa para que seja considerada válida.  
  *Default:* 10

---

### 7.4 Parâmetros Cosméticos
- **`mot20`** *(bool)*  
  Ajusta parâmetros para datasets tipo MOT20 (objetos muito próximos e densos).

---

## Exemplo de arquivo `bytetrack.yaml`

```yaml
# Ultralytics ByteTrack config
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
min_box_area: 10
mot20: False
```

---

## 8. Exemplo de Uso Completo

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.track(
    source="video.mp4",
    tracker="bytetrack.yaml",
    conf=0.25,
    iou=0.7,
    show=True,
    save=True,
    persist=True,
    device="0",
    imgsz=640
)
```

---

# 9. Resumo Geral
Este manual apresenta:
- todos os argumentos disponíveis para `model.track()` no YOLOv8,
- explicações por categoria,
- parâmetros completos do ByteTrack aceitos no Ultralytics,
- exemplo final de uso.

A estrutura modular facilita incorporar este conteúdo em documentação técnica, aplicações de IA, sistemas de videomonitoramento e pipelines avançados de visão computacional.

