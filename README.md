# SoCV Final Project: 2026 CAD Contest B: Regression Failure Bucketing

---

Team ID: cadb1053  
Team Name: bububusc  
Team Members:  
B11901047 郭祐嘉  
B11901043 張庭碩  
B11901112 卜紹秦

---
Install dependencies:
```bash
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