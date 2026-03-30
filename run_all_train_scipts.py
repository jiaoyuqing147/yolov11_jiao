import subprocess
import time
from datetime import datetime

# =========================
# 你要顺序运行的训练脚本
# =========================
scripts = [
    # "train_GTSDB.py",
    "train_GTSDB_deter.py",
    "train_GTSDB_deter1.py",
    # "train_tt100k.py",
    # "train_tt100k_deter.py",
    # "train_tt100k_deter1.py",
]

# =========================
# 配置
# =========================
WAIT_SECONDS = 100          # 两个任务之间等待 5 分钟
CURRENT_JOB_FILE = "current_job.txt"


def write_current_job(job_name: str):
    """写入当前正在运行的任务名，供 watchdog 读取。"""
    with open(CURRENT_JOB_FILE, "w", encoding="utf-8") as f:
        f.write(job_name)


def clear_current_job():
    """清空当前任务状态。"""
    with open(CURRENT_JOB_FILE, "w", encoding="utf-8") as f:
        f.write("")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    print(f"🚀 Batch training started at {now_str()}")

    for i, script in enumerate(scripts):
        print(f"\n========== Running [{script}] ==========\n")

        # 告诉 watchdog 当前任务是谁
        write_current_job(script)

        log_name = f"log{i}.txt"

        with open(log_name, "w", encoding="utf-8") as logfile:
            logfile.write(f"[{now_str()}] START {script}\n")
            logfile.flush()

            process = subprocess.Popen(
                ["python", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            # 实时读取输出
            for line in process.stdout:
                print(line, end="")
                logfile.write(line)
                logfile.flush()

            process.wait()
            return_code = process.returncode

            logfile.write(f"\n[{now_str()}] END {script} | return code = {return_code}\n")
            logfile.flush()

        if return_code == 0:
            print(f"\n✅ [{script}] finished normally, log saved to {log_name}\n")
        else:
            print(f"\n⚠️ [{script}] exited abnormally with return code {return_code}, log saved to {log_name}\n")

        # 当前任务结束，清空状态
        clear_current_job()

        # 如果不是最后一个任务，等待 5 分钟
        if i < len(scripts) - 1:
            print(f"⏳ Waiting {WAIT_SECONDS} seconds before starting next script...\n")
            time.sleep(WAIT_SECONDS)

    print(f"🎉 All training jobs completed at {now_str()}")