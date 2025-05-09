README.md
## 🧪 Setup & Run Instructions

### ✅ Step 1: Create Virtual Environment

```bash
python -m venv venv
```

---

### ✅ Step 2: Activate the Virtual Environment

**On macOS/Linux:**

```bash
source venv/bin/activate
```

**On Windows:**

```bash
venv\Scripts\activate
```

---

### ✅ Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

---

### ✅ Step 4: Run Neural Network

```bash
cd neural_network
python nn.py
```

---

### ✅ Step 5: Run k-NN Algorithm

```bash
cd knn_algorithm
python knn.py
```

---

### ✅ Step 6: Run Random Forest

```bash
cd random_forest
python tree.py
```

### ✅ Step 7: Run Decision Tree

```bash
cd decision_tree
python dt.py
```

### ✅ Step 8: Change Settings in Each Script.
# Dataset name, Hyper Parameter, Stopping Criteria are located on the top of the all script. Please change to try different dataset or other settings. Below is random_forest/tree.py
```bash
DATASET_NAME="heart_disease"
K_FOLD_SIZE=10
MAX_DEPTH = 5
MIN_INFO_GAIN = 1e-5
```