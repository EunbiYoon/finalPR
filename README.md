# Step 1: Create virtual environment
python -m venv venv

# Step 2: Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Step 3: Install Library
pip install -r requirements.txt

# Step 4 : Run Script
## run neural network
cd neurnal_network
python nn.py

## run the knn algorithm
cd knn_algorithm
python knn.py