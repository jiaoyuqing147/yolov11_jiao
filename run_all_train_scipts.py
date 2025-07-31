import subprocess
import time  # ✅ 添加这一行

scripts = [
    "train_tt100k_pro_chu.py",
    # "train_tt100k_pro_chu1.py",
    "train_tt100k_pro_chu2.py",
    "train_tt100k_pro_chu3.py",
    # "train_tt100k_pro_chu4.py",
    # "train_tt100k_pro_chu5.py",
    # "train_tt100k_pro_chu6.py"
]

for i, script in enumerate(scripts):
    print(f"\n========== Running [{script}] ==========\n")

    with open(f"log{i}.txt", "w", encoding="utf-8") as logfile:
        process = subprocess.Popen(
            ["python", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",       # ✅ 解决乱码问题
            errors="replace"        # ✅ 替代不可解字符
        )

        for line in process.stdout:
            print(line, end="")
            logfile.write(line)

        process.wait()
        print(f"\n✅ [{script}] finished, log saved to log{i}.txt\n")

    if i < len(scripts) - 1:
        print("⏳ Waiting 5 minutes before starting next script...\n")
        time.sleep(300)  # ⏸️ 暂停 5 分钟（300秒）

print("🎉 All training jobs completed.")