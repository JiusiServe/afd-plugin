import base64
s = """import json,urllib.request
req=urllib.request.Request('http://127.0.0.1:8900/v1/chat/completions',data=json.dumps({"model":"dsv4-async","messages":[{"role":"user","content":"1+1=? Reply with just the number."}],"max_tokens":32,"temperature":0}).encode(),headers={'Content-Type':'application/json'})
try:
    r=urllib.request.urlopen(req,timeout=120)
    print('HTTP',r.status)
    print(r.read().decode())
except Exception as e:
    print('ERR',repr(e))
"""
print(base64.b64encode(s.encode()).decode())
