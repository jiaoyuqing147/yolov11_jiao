import subprocess

scripts = [
    # "train_tt100k_pro_myxlab.py",
    # "train_tt100k_pro_myxlab1.py",
    # "train_tt100k_pro_myxlab2.py",
    # "train_tt100k_pro_myxlab21.py",
    # "train_tt100k_pro_myxlab3.py",
    # "train_tt100k_pro_myxlab4.py"
    # "train_tt100k_pro_myxlab5.py"
    "train_tt100k_pro_myxlab21.py",
    "train_tt100k_pro_myxlab22.py"
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

print("🎉 All training jobs completed.")
