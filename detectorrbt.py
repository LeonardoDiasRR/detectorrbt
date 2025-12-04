# built-in
import time
import threading
import os
import traceback
from dotenv import load_dotenv

# 3d party
import yaml
import shutil
import torch
import logging
from ultralytics import YOLO

# local
from findface_multi import FindfaceMulti
from findface_adapter import obter_lista_cameras_virtuais_findface

# Configura logging ANTES de importar outros módulos
log_file = os.path.join(os.path.dirname(__file__), "detectorrbt.log")
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.FileHandler(log_file, mode='w', encoding='utf-8'),
		logging.StreamHandler()  # Mantém output no console também
	],
	force=True  # Força reconfiguração mesmo se já foi chamado
)

from bytetrack_detector import ByteTrackDetector


def main():
	logger = logging.getLogger(__name__)
	
	# Limpa o diretório de imagens antes de iniciar
	imagens_dir = os.path.join(os.path.dirname(__file__), "imagens")
	if os.path.exists(imagens_dir):
		try:
			shutil.rmtree(imagens_dir)
			logger.info(f"Diretório '{imagens_dir}' removido.")
		except Exception as e:
			logger.warning(f"Não foi possível remover '{imagens_dir}': {e}")
	
	# Recria o diretório vazio
	os.makedirs(imagens_dir, exist_ok=True)
	logger.info(f"Diretório '{imagens_dir}' criado.")
	
	cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
	if not os.path.exists(cfg_path):
		cfg_path = "config.yaml"

	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f) or {}

	gpu_index = cfg.get("gpu_index", 0)
	device_type = f"cuda:{gpu_index}" if (torch is not None and torch.cuda.is_available()) else "cpu"

	# choose batch size based on GPU availability
	if device_type.startswith("cuda"):
		batch_size = cfg.get("gpu_batch_size", 32)
	else:
		batch_size = cfg.get("cpu_batch_size", 4)

	show = cfg.get("show", True)

	cameras = cfg.get("cameras", [])
	processors = []

	# prepare a YOLO model instance (shared) if ultralytics is available
	try:
		weights = cfg.get("yolo_weights") or os.path.join(os.path.dirname(__file__), "yolov8n-face.pt")
		if not os.path.exists(weights):
			weights = os.path.join(os.getcwd(), "yolov8n-face.pt")
		yolo_model = None
		if YOLO is not None:
			try:
				yolo_model = YOLO(weights)
				logger.info(f"Modelo YOLO carregado: {weights}")
			except Exception as e:
				logger.warning(f"Não foi possível carregar o modelo YOLO '{weights}': {e}")
				yolo_model = None
	except Exception:
		yolo_model = None

	
	cameras_ff = obter_lista_cameras_virtuais_findface(ff, prefixo=cfg.get("findface_camera_prefix", "TESTE")) + cameras
	for cam in cameras_ff:
		cam_id = cam.get("id")
		cam_name = cam.get("name", f"camera_{cam_id}")
		# aceitar chave `url` ou compatibilidade retroativa com `source`
		cam_url = cam.get("url")
		camera_token = cam.get("token")

		if cam_id is None or cam_url is None:
			logger.warning(f"Câmera inválida no config: {cam}")
			continue

		p = ByteTrackDetector(
			camera_id=cam_id,
			camera_name=cam_name,
			camera_token=camera_token,
			source=cam_url,
			yolo_model=yolo_model,
			batch=batch_size,
			show=show,
			)

		# Cria e inicia thread explicitamente
		thread = threading.Thread(target=p.start, daemon=True)
		thread.start()
		processors.append((p, thread))

	try:
		# keep main thread alive
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		logger.info("Interrupção recebida, finalizando...")
		for p, thread in processors:
			p.stop()
		for p, thread in processors:
			thread.join(timeout=5)


if __name__ == "__main__":
	# Carrega variáveis de ambiente do arquivo .env
	load_dotenv()
	
	try:
		ff = FindfaceMulti(		
			url_base=os.environ["FINDFACE_URL"],
			user=os.environ["FINDFACE_USER"],
			password=os.environ["FINDFACE_PASSWORD"],
			uuid=os.environ["FINDFACE_UUID"]
		)

		main()
		
	except KeyError as e:
		print(traceback.format_exc())
	finally:
		ff.logout()