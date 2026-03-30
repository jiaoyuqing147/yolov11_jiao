import time
import psutil
import subprocess
import os
#这个要和run_all_train_scipts配合，实时监控现在运行的哪个进程，如果服务求卡死，几分钟后就把此程序杀掉，防止它影响服务器。
CHECK_INTERVAL = 30
TIME_THRESHOLD = 300
CPU_THRESHOLD = 5
GPU_THRESHOLD = 5
JOB_FILE = "current_job.txt"

low_usage_duration = 0


def get_gpu_usage():
    try:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
            shell=True
        )
        values = result.decode().strip().split("\n")
        values = [int(v) for v in values if v.strip()]
        return max(values) if values else 0
    except Exception:
        return 0


def read_current_job():
    if not os.path.exists(JOB_FILE):
        return ""
    try:
        with open(JOB_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def find_processes_by_script(script_name):
    matches = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = p.info.get("cmdline") or []
            cmd = " ".join(cmdline)
            if script_name and script_name in cmd:
                matches.append(p)
        except Exception:
            continue
    return matches


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        for child in children:
            try:
                child.kill()
            except Exception:
                pass

        try:
            parent.kill()
        except Exception:
            pass

        print(f"🔥 Killed PID={pid} and its children")
    except Exception as e:
        print(f"Kill failed: {e}")


print("🛡️ Watchdog started...")

for p in psutil.process_iter():
    try:
        p.cpu_percent(interval=0)
    except Exception:
        pass

while True:
    current_job = read_current_job()

    if not current_job:
        print("❗ No current job")
        low_usage_duration = 0
        time.sleep(CHECK_INTERVAL)
        continue

    procs = find_processes_by_script(current_job)

    if not procs:
        print(f"❗ Job file says [{current_job}] but no process found")
        low_usage_duration = 0
        time.sleep(CHECK_INTERVAL)
        continue

    gpu = get_gpu_usage()
    total_cpu = 0.0

    print(f"\n📌 Current job: {current_job}")
    for p in procs:
        try:
            cpu = p.cpu_percent(interval=0.1)
            total_cpu += cpu
            print(f"PID={p.pid} | CPU={cpu:.1f}% | CMD={' '.join(p.cmdline())}")
        except Exception:
            pass

    print(f"📊 Total CPU={total_cpu:.1f}% | GPU={gpu}%")

    if total_cpu < CPU_THRESHOLD and gpu < GPU_THRESHOLD:
        low_usage_duration += CHECK_INTERVAL
        print(f"⚠️ Low usage for {low_usage_duration}s")

        if low_usage_duration >= TIME_THRESHOLD:
            print(f"💀 Detected possible hang in [{current_job}]! Killing it...")
            for p in procs:
                try:
                    kill_process_tree(p.pid)
                except Exception:
                    pass
            low_usage_duration = 0
    else:
        low_usage_duration = 0

    time.sleep(CHECK_INTERVAL)