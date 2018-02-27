import requests
import sys
from playsound import playsound

sound_names = ["air conditioner", "car horn", "children playing", "dog bark", "drilling", "engine idling","gun shot", "jackhammer", "siren", "street music"]


playsound(sys.argv[1])
url = 'http://127.0.0.1:5000/upload'
files = {'file': open(sys.argv[1], 'rb')}
r = requests.post(url, files=files)
print(r.text)
index = int(r.text)
if index < 0 or index >= len(sound_names):
    print("Error, index out of range")
else:
    print(sound_names[index])

