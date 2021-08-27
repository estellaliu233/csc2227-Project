import requests
import pandas as pd
import numpy as np
import json


url = 'http://0.0.0.0:5000/api/'



buckets = [[1.2 for col in range(12)] for row in range(1)]
j_data = json.dumps(buckets)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)