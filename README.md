**NumCompute** : A modular, production‑grade scientific computing toolkit using plain Python + NumPy

**Project Goal** : To design and implement an end-to-end ML-style framework from scratch without external ML libraries, covering: Data ingestion, Preprocessing, Algorithms (search, ranking), Statistics, Evaluation metrics, Optimisation (gradients), Pipeline abstraction, Benchmarking

**Features** **Includes :** 
1. **Data I/O (io.py)**
CSV reader with mixed data support (numeric + categorical)
Handles missing values (NaN normalization)
Supports flexible dtype and streaming-friendly design

2. **Preprocessing (preprocessing.py)**
Imputer (numeric mean, categorical mode)
StandardScaler (Z-score normalization)
OneHotEncoder (categorical → numeric)
Works seamlessly with mixed-type datasets
3. **Algorithms (sort_search.py)**
Top-K selection using np.argpartition → O(n)
Binary Search (returns index + insertion point)
4. **Ranking (rank.py)**
Tie-aware ranking:
average
dense
ordinal
Percentile computation with interpolation:
linear, lower, higher, midpoint
5. **Statistics (stats.py)**
Mean, Std, Min, Max (NaN-safe)
Histogram computation
Quantiles (built on percentile)
6. **Metrics (metrics.py)**
Classification:
Accuracy, Precision, Recall, F1 Score
Regression:
Mean Squared Error (MSE)
Confusion matrix support
7. **Optimisation (optim.py)**
Finite difference gradients:
central difference (default)
forward difference
Jacobian computation for vector-valued functions
8. **Pipeline (pipeline.py)**
Modular transformation chaining (like scikit-learn pipelines)
Consistent API:
fit()
transform()
fit_transform()
9. **Benchmarking (benchmarking.py)**
Compare vectorised vs loop implementations
Reproducible timing utilities
Performance analysis output

 **Installation**
```
git clone https://github.com/ns-rg/NumCompute.git
cd NumCompute
```
No external dependencies required 

**Quick Start**
Run the demo notebook:
```
cd demo
jupyter notebook quickstart.ipynb
```

**Example Usage**
1. Pipeline Example
```py
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import Imputer, StandardScaler
pipe = Pipeline([
    ("impute", Imputer()),
    ("scale", StandardScaler())
])
X_out = pipe.fit_transform(X)
```
2. Top-K Example
```py
from numcompute.sort_search import topk
import numpy as np

arr = np.array([10, 5, 20, 15])
vals, idx = topk(arr, 2)
```
3. Gradient Example
```py
from numcompute.optim import grad
import numpy as np

def f(x):
    return x[0]**2 + x[1]**2

grad(f, np.array([2.0, 3.0]))
```
 **Performance**
Vectorised NumPy operations significantly outperform Python loops:
```
Operation			Loop Time			Vectorised Time				Speedup
Sum (1M elements)	~0.045s				~0.001s						~40x
```

Vectorisation leverages optimized C-level execution in NumPy, reducing Python overhead.

**Design Highlights**
1. Vectorisation-first approach for performance
2. Robust handling of:
	NaNs
	mixed data types
	edge cases
3. Clean and consistent API across modules
4. Modular architecture for extensibility
5. Numerical stability:
epsilon handling
central difference gradients
safe conversions

**Testing**
Comprehensive unit tests (20+ cases) covering:
1. Edge cases:
empty arrays
NaNs
duplicates/ties
invalid inputs
Functional correctness across all modules

Run tests:

```py
python -m tests.test_preprocessing
python -m tests.test_stats
python -m tests.test_optim
python -m tests.test_rank
```
**Project Structure**
```
NumCompute/
├── numcompute/
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── benchmarking.py
│   └── utils.py
├── tests/
├── demo/
│   └── quickstart.ipynb
├── README.md
└── setup.py
```
**Team TaskForce Contribution**
The project was developed collaboratively, with all team members contributing across multiple modules. While responsibilities were initially divided to improve efficiency, significant overlap existed through code reviews, debugging, testing, and integration.

All members contributed to:
-   debugging and fixing edge cases
-   writing and refining unit tests
-   improving numerical stability and performance
-   integrating modules into a unified pipeline
