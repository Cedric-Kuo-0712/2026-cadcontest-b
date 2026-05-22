# SoCV Final Project: 2026 CAD Contest B: Regression Failure Bucketing

---

Team ID: cadb1053  
Team Name: bububusc  
Team Members:  
B11901047 郭祐嘉  tankkuo0712@gmail.com  
B11901043 張庭碩  timmychang104@gmail.com  
B11901112 卜紹秦  pushaochin@gmail.com  

---
Install dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

How to run:
```bash
python src/regr_fail_bucketing.py \
    --input B_samples_20260516/problem/benchmark_set_1/input.csv \
    --output output.csv \
    --k 2
```
Evaluation:
```bash
python3 eval.py \
  --output output.csv \
  --golden B_samples_20260516/problem/benchmark_set_1/golden.csv
```

# 支線任務 - 生成測資

打算自己用 UVM 生成測資觀察一下是否可以做更多統計分析，但是如果是打算在同一個design當中插入多個bug, 然後每個輸出的sim.log代表他遇到某個bug, 這樣我們也不知道他實際上是哪個bug(就變回我們要做unsupersived learning的任務)。或是可以只插一個bug, 看他在分類上是否都可以被分到同一類。