import sys
test = False
dev = False
best = {"test_ave":0.0,"dev_ave":0.0}
start = False
while True:
    line = sys.stdin.readline()
    if not line:break
    line = line.strip()
    if len(line) <= 0:
        break
 
while True:
    line = sys.stdin.readline()
    if not line:break
    line = line.strip()
    if line.startswith("Pretrain"):
        epoch = line.split(" ")[-1]
    elif line.startswith("DEV"):
        dev = True
    elif line.startswith("TEST"):
        test = True
    elif line.startswith("Average"):
        if test:
            test_score = float(line.split(" ")[-1])
        elif dev:
            dev_score = float(line.split(" ")[-1]) 
    elif len(line) == 0:
        if dev_score >= best["dev_ave"]:
            best["dev_ave"] = dev_score
            best["test_ave"] = test_score
            best["epoch"] = epoch
        dev = False
        test = False
        start = False
print best["epoch"],best["dev_ave"],best["test_ave"]
