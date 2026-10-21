import subprocess
from pathlib import Path

key = Path.home() / ".ssh" / "forgeml_amd"
cmd = [
    "ssh", "-i", str(key),
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
    "root@165.245.135.243",
    "echo FORGEML_SSH_OK",
]

for attempt in range(1, 6):
    print(f"\n--- SSH attempt {attempt}/5 ---", flush=True)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=20
        )
        print("Exit code:", result.returncode)
        print("STDOUT:", result.stdout.strip())
        print("STDERR:", result.stderr.strip())
    except Exception as exc:
        print(type(exc).__name__ + ":", exc)
