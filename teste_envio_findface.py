# built-in
from dotenv import load_dotenv
import traceback
import os
import json

# local
from findface_multi import FindfaceMulti
from config_loader import CONFIG


def main(ff: FindfaceMulti):
    arquivo = r'C:\Users\leonardo.lad\Pictures\Foto001.webp'

    camera_id = 252
    camera_token = "fde4417d79004e399907ff07bb2810dc"

    resultado = ff.add_face_event(
		token=camera_token, 
		fullframe=arquivo, 
		camera=camera_id,
		roi=[348, 243, 460, 382],
        mf_selector="all"
		)

    print(json.dumps(resultado, indent=4, ensure_ascii=False))

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
		main(ff)
		
	except KeyError as e:
			print(traceback.format_exc())
	finally:
		ff.logout()