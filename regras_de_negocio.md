# RESUMO
A aplicação 'detectorrbt.py' deve assistir a vários fluxos de vídeos RTSP simultaneamente, em cada um, realizar tracks das faces, selecionar a melhor face de cada track baseado em um algoritmo que avalia a qualidade da face, e salvar no diretório 'imagens' o frame completo com o evento de detecção da melhor face do track com o bbox desenhado. O nome do arquivo salvo deve seguir o padrão 'Camera_<camera_id>-<camera_name>_Track_<track_id>_<timestamp_do_frame>.jpg'.

# MAIN
A função 'main()' em 'detectorrbt.py', deve:
1. Ler o arquivo de configuração 'config.yaml'
2. Carregar o modelo YOLO especificado no config
3. Ler a seção 'cameras' do arquivo de configuração
4. Para cada câmera, iniciar o processamento instanciando CameraProcessor em uma thread independente

## Inicialização dos atributos dinâmicos

### device_type
Na inicialização, deve ser verificado se o sistema possui suporte GPU CUDA. O atributo 'device_type' deve ser setado conforme:

```python
device_type = f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
```

O valor de 'gpu_index' consta no atributo 'gpu_index' do arquivo de configuração 'config.yaml'

### batch_size
Em seguida, deve ler no arquivo de configuração 'config.yaml':
- A seção 'gpu_batch', se houver suporte à GPU, OU
- A seção 'cpu_batch', se não houver suporte à GPU

Setar o atributo dinâmico 'batch_size' com o respectivo valor lido.

# CAMERA PROCESSOR
Crie o arquivo camera_processor.py e nele crie a classe CameraProcessor. 

## Parâmetros de inicialização obrigatórios:
- camera_id (int)
- camera_name (str)
- source (str)  # URL RTSP
- yolo_model (YOLO)
- device_type (str)
- batch_size (int)
- config (dict)  # Dicionário com as configurações da aplicação lidas de 'config.yaml'

## Comportamento da classe:
A classe CameraProcessor deve:
1. Executar detecções de objetos na fonte informada (source)
2. Utilizar o modelo YOLO informado (yolo_model)
3. Utilizar o tipo de dispositivo especificado (device_type)
4. **Obrigatoriamente** utilizar o algoritmo ByteTrack do Ultralytics para rastreamento
5. **Obrigatoriamente** utilizar inferência em lote (batch_size)
6. Para cada track ativo, armazenar todos os frames e suas respectivas qualidades de face

## Detecção de tracks finalizados:
Um track é considerado finalizado quando:
- O objeto não é mais detectado por N frames consecutivos (N definido em config.yaml como 'max_frames_lost')
- OU quando o processamento do stream é interrompido

## Tratamento de erros:
- Se a conexão RTSP falhar, tentar reconectar automaticamente a cada X segundos (definido em config.yaml como 'reconnect_interval')
- Registrar todos os erros em log
- Se após Y tentativas (definido em config.yaml como 'max_reconnect_attempts') não conseguir conectar, encerrar a thread

# ESCOLHA DA MELHOR FACE EM CADA TRACK
Sempre que um track for finalizado, deve ser escolhida a melhor face do track através dos seguintes passos:

1. Para cada frame armazenado do track, calcular o score de qualidade utilizando a função 'get_face_quality_score()' de 'face_quality_score.py'
2. Selecionar o frame com o maior score de qualidade
3. Desenhar o bbox da face no frame selecionado:
   - Cor: Verde (0, 255, 0)
   - Espessura: 2 pixels
   - Incluir label com o track_id e score de qualidade

## Salvamento da imagem:
O frame completo (não apenas o crop da face) com o bbox desenhado deve ser salvo em './imagens' seguindo o padrão:
- Nome: 'Camera_<camera_id>-<camera_name>_Track_<track_id>_<timestamp_do_frame>.jpg'
- Formato: JPEG
- Qualidade: 95

# ESTRUTURA DO CONFIG.YAML ESPERADA
```yaml
gpu_index: 0
gpu_batch: 16
cpu_batch: 4
max_frames_lost: 30
reconnect_interval: 5
max_reconnect_attempts: 10
yolo_model_path: "path/to/yolov8n.pt"

cameras:
  - id: 1
    name: "Entrada"
    source: "rtsp://..."
  - id: 2
    name: "Saida"
    source: "rtsp://..."
```