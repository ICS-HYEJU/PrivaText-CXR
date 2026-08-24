import json


with open("/Model/Diffusion/origin_forward_log.json") as f:
    a = json.load(f)

with open("/Model/Diffusion/forward_log.json") as f:
    b = json.load(f)

def compare(a, b, path=""):
    if type(a) != type(b):
        print(f"[TYPE DIFF] {path}: {type(a)} != {type(b)}")
        return

    if isinstance(a, dict):
        for k in a.keys():
            if k not in b:
                print(f"[MISSING] {path}.{k} not in B")
            else:
                compare(a[k], b[k], path + "." + k)

        for k in b.keys():
            if k not in a:
                print(f"[MISSING] {path}.{k} not in A")

    elif isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + f"[{i}]")

        if len(a) != len(b):
            print(f"[LEN DIFF] {path}: {len(a)} != {len(b)}")

    else:
        if a != b:
            print(f"[DIFF] {path}: {a} != {b}")
    print(f"[DONE]")
# ½ÇÇà
compare(a, b)