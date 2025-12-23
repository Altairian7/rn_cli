import json

with open("gnss_1.json") as f:
    txt = f.read()

if not txt.strip().endswith("]"):
    txt = txt.rstrip(", \n") + "]"

# try parsing; if it fails, cut back to last full object
while True:
    try:
        data = json.loads(txt)
        break
    except json.JSONDecodeError as e:
        txt = txt[:e.pos].rsplit("{",1)[0].rstrip(", \n") + "]"

with open("gnss_1_fixed.json","w") as f:
    f.write(txt)
print("Fixed saved to gnss_1_fixed.json")
