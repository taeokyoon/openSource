import os, time, datetime, pathlib

interval = int(os.getenv("MSG_INTERVAL", "2"))
pathlib.Path("/data").mkdir(parents=True, exist_ok=True)
log = "/data/messages.log"

i = 0
while True:
    i += 1
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] hello #{i}\n"
    with open(log, "a") as f:
        f.write(line)
    print(f"writer> appended: {line.strip()}")
    time.sleep(interval)
