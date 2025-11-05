import time, os

log = "/data/messages.log"
print("reader> waiting for messages.log ...")
while not os.path.exists(log):
    time.sleep(1)

seen = 0
print("reader> tailing /data/messages.log")
while True:
    with open(log, "r") as f:
        lines = f.readlines()
    for line in lines[seen:]:
        print("reader>", line.strip())
    seen = len(lines)
    time.sleep(1)
