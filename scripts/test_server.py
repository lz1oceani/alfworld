import requests
from IPython import embed
from base64 import b64decode, b64encode
from pickle import loads, dumps


env_url = "http://127.0.0.1:3000"
requests.post(env_url + "/set_environment", json={"env_type": "visual"}).text
text = requests.post(env_url + "/reset", json={}).text
text = eval(text)
text = b64decode(text)
[obs, image], infos = loads(text)

embed()

