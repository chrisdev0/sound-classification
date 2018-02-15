import requests
import sys
from playsound import playsound

playsound(sys.argv[1])
url = 'http://127.0.0.1:5000/upload'
files = {'file': open(sys.argv[1], 'rb')}
r = requests.post(url, files=files)
print(r.text)

