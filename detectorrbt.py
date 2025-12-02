import yaml
import time
import threading
import os
import shutil
import torch
from ultralytics import YOLO

from camera_processor import CameraProcessor
from bytetrck_detector import ByteTrackDetector


def main():
	# Limpa o diretório de imagens antes de iniciar
	imagens_dir = os.path.join(os.path.dirname(__file__), "imagens")
	if os.path.exists(imagens_dir):
		try:
			shutil.rmtree(imagens_dir)
			print(f"[main] Diretório '{imagens_dir}' removido.")
		except Exception as e:
			print(f"[main] Aviso: não foi possível remover '{imagens_dir}': {e}")
	
	# Recria o diretório vazio
	os.makedirs(imagens_dir, exist_ok=True)
	print(f"[main] Diretório '{imagens_dir}' criado.")
	
	cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
	if not os.path.exists(cfg_path):
		cfg_path = "config.yaml"

	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f) or {}

	gpu_index = cfg.get("gpu_index", 0)
	device_type = f"cuda:{gpu_index}" if (torch is not None and torch.cuda.is_available()) else "cpu"

	# choose batch size based on GPU availability
	if device_type.startswith("cuda"):
		batch_size = cfg.get("gpu_batch", 32)
	else:
		batch_size = cfg.get("cpu_batch", 4)

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
			except Exception as e:
				print(f"[main] warning: unable to load YOLO weights '{weights}': {e}")
				yolo_model = None
	except Exception:
		yolo_model = None

	for cam in cameras:
		cam_id = cam.get("id")
		cam_name = cam.get("name", f"camera_{cam_id}")
		# aceitar chave `url` ou compatibilidade retroativa com `source`
		cam_url = cam.get("url") or cam.get("source")
		if cam_id is None or cam_url is None:
			print(f"[main] câmera inválida no config: {cam}")
			continue

		p = ByteTrackDetector(
			camera_id=cam_id,
			camera_name=cam_name,
			source=cam_url,
			yolo_model=yolo_model,
			batch=batch_size,
			show=show,
			)

		# p = CameraProcessor(camera_id=cam_id, camera_name=cam_name, source=cam_url,
		# 			yolo_model=yolo_model, device_type=device_type, batch_size=batch_size, config=cfg)

		p.start()
		processors.append(p)

	try:
		# keep main thread alive
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		print("[main] recebida interrupção, finalizando...")
		for p in processors:
			p.stop()
		for p in processors:
			p.join(timeout=5)


if __name__ == "__main__":
	main()

