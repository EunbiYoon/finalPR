📝 Homework Overview & Execution Guide

---

📦 HOMEWORK 1 – kNN & Decision Tree

Contents:
datasets/
  ├── car.csv       # Car Evaluation Dataset (ZIP 제공)
  ├── iris.csv      # Iris Dataset (추가 성능 비교용)
  ├── wdbc.csv      # Wisconsin Breast Cancer Dataset (ZIP 제공)
  ├── wine.csv      # Wine Dataset (추가 성능 비교용)
HW1.py              # 주 실행 파일
README.txt          # 실행 안내 문서

Execution Format:
python HW1.py <mode> <dataset> <hw> --k <k> --normalize <0|1> --func <func> --limit <0|1>

Parameters:
- mode (required): knn, knearestneighbors, decisiontree, dt
- dataset: wdbc, car, or custom CSV path
- hw: 0 (custom run) or 1 (homework Q1/Q2 전용 실행)
- --k: KNN의 k 값 (default: 3)
- --normalize: 정규화 여부 (default: 1)
- --func: 정보 이득 계산 방식 (entropy, gini)
- --limit: 트리 깊이 제한 (default: 0 → 무제한)

Sample Commands:
python HW1.py knn wdbc 1
python HW1.py decisiontree car 1
python HW1.py knn wdbc 0 --k 5 --normalize 0
python HW1.py decisiontree car 0 --func entropy --limit 1

---

📦 HOMEWORK 2 – Naive Bayes

Contents:
COMPSCI_589___HW2.pdf       # 제출용 PDF 보고서
figure-q3.png               # Q3 결과 그래프
hw2.py                      # 주 실행 파일
utils.py                    # CSV 및 평가 함수 모듈
README.txt                  # 실행 안내 문서
train-*.csv, test-*.csv     # 긍정/부정 리뷰 학습/테스트 세트

Execution Format:
python main.py --runs <n> --question <n|all> --debug <0|1>

Parameters:
- --runs: 시뮬레이션 반복 횟수 (default: 1)
- --question: 실행할 과제 문항 (1, 2, 3, 4, 6, or all)
- --debug: 디버깅 출력 여부

Sample Commands:
python main.py
python main.py --runs 10 --question 3
python main.py --runs 5 --question 1 --debug 1

---

📦 HOMEWORK 3 – Random Forest & Stratified K-Fold

Contents:
datasets/
  ├── wdbc.csv      # Main dataset 1
  ├── loan.csv      # Main dataset 2
  ├── raisin.csv    # Bonus dataset 1
  ├── titanic.csv   # Bonus dataset 2
hw3.py              # 주 실행 파일
README.txt          # 실행 안내 문서

Execution Format:
python hw3.py <data> <ntrees> --k <k> --mode <mode> --msfs <msfs> --md <md> --random_state <random_state>

Parameters:
- data (required): wdbc, loan, raisin, titanic, or path to CSV
- ntrees (required): Random Forest의 트리 개수
- --k: K-fold 수 (default: 5)
- --mode: 정보 이득 계산 방법 ('entropy' or 'gini')
- --msfs: 최소 샘플 수 for split (default: -1 → 무제한)
- --md: 최대 트리 깊이 (default: -1 → 무제한)
- --random_state: 시드 값

Sample Commands:
python hw3.py wdbc 10
python hw3.py loan 50 --msfs 10 --md 3
python hw3.py datasets/xyz.csv 5 --k 3 --md 4 --random_state 1000

---

📦 HOMEWORK 4 – Neural Networks

Contents:
datasets/
  ├── wdbc.csv      # Main dataset 1
  ├── loan.csv      # Main dataset 2
  ├── raisin.csv    # Bonus dataset 1
  ├── titanic.csv   # Bonus dataset 2
hw4.py              # 주 실행 파일
hw4_source.pdf      # HW4 보고서 PDF
README.txt          # 실행 안내 문서

Execution Format:
python hw4.py <mode> --lc_exp <lc_exp> --layer_sizes <layer_sizes> --epochs <epochs> --lamb <lamb> --lr <lr> --verbose <verbose>

Parameters:
- mode (required): verify, wdbc, loan, raisin, titanic, or path to CSV
- --lc_exp: 학습 곡선 생성 여부 (0 or 1)
- --layer_sizes: 히든 및 출력층 구조 (ex. 64 32 1)
- --epochs: 반복 학습 수 (default: 2000)
- --lamb: 정규화 파라미터 λ (default: 0.01)
- --lr: 학습률 (default: 0.5)
- --verbose: 디버깅 정보 출력 (0 or 1)

Sample Commands:
python hw4.py verify
python hw4.py wdbc --epochs 1738 --layer_sizes 5 1 --lamb 0 --lr 0.2
python hw4.py loan --lc_exp 1
python hw4.py datasets/xyz.csv --lamb 0.05 --lr 0.15