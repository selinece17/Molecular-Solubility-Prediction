# Molecular Solubility Prediction with Machine Learning

 Project Overview

This project implements a complete pipeline for predicting molecular solubility using the ESOL (Delaney) dataset from MoleculeNet. The implementation demonstrates traditional machine learning with molecular fingerprints and explores modern Graph Neural Network (GNN) approaches for molecular property prediction.

### Key Features
-  **ESOL Dataset**: 1,128 molecules with measured aqueous solubility
-  **Molecular Featurization**: Morgan fingerprints and graph representations
-  **Random Forest Baseline**: Traditional ML approach achieving R² = 0.71
-  **Graph Neural Networks**: Exploring GNN architectures for molecular graphs
-  **Comprehensive Evaluation**: Multiple metrics and visualizations
-  **Drug Predictions**: Inference on common pharmaceutical compounds

---


##  Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/solubility-prediction.git
cd solubility-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook solubility-prediction__1_.ipynb
```

---

##  Dataset

### ESOL (Delaney) Dataset
- **Source**: MoleculeNet benchmark dataset
- **Size**: 1,128 molecules
- **Target**: Measured log solubility in moles per litre
- **Format**: SMILES strings with experimental measurements

### Dataset Statistics

```
Total molecules: 1,128
Solubility range: -11.6 to +1.58 log(S)
Mean solubility: -3.05 ± 2.10 log(S)
```

### Solubility Distribution

| Percentile | Log Solubility |
|------------|----------------|
| 25%        | -4.32          |
| 50% (Median) | -2.86        |
| 75%        | -1.60          |

**Interpretation:**
- **Highly Soluble**: log(S) > -2
- **Moderately Soluble**: -4 < log(S) < -2
- **Poorly Soluble**: log(S) < -4

---

##  Methodology

### Pipeline Architecture

```
SMILES Strings → Molecular Features → ML Models → Solubility Predictions
                 (Fingerprints/Graphs)   (RF/GNN)
```

### 1. Molecular Featurization

**Approach 1: Morgan Fingerprints (Random Forest)**
- **Circular fingerprints** (ECFP4 equivalent)
- **Radius**: 2
- **Bits**: 2,048
- Captures molecular substructures and chemical patterns

**Approach 2: Graph Representation (GNN)**
- **Nodes**: Atoms with features
  - Atom type (one-hot: C, N, O, S, F, P, Cl, Br, I)
  - Degree, formal charge, radical electrons
  - Hybridization, aromaticity, hydrogen count
- **Edges**: Chemical bonds (bidirectional)
- **Feature dimension**: 16 per atom

### 2. Machine Learning Models

#### Random Forest Regressor
- **Architecture**: Ensemble of 100 decision trees
- **Input**: 2,048-dimensional Morgan fingerprints
- **Output**: Continuous solubility value
- **Advantages**: Interpretable, robust, fast training

#### Graph Neural Network (Experimental)
- **Architecture**: Graph Convolutional Network (GCN)
- **Layers**: Multiple GCN layers with ReLU activation
- **Pooling**: Global mean pooling over nodes
- **Output**: Graph-level prediction
- **Advantages**: Learns from molecular structure directly

### 3. Train-Test Split
- **Training**: 80% (902 molecules)
- **Testing**: 20% (226 molecules)
- **Random seed**: 42 (reproducibility)

---

##  Results

### Random Forest Performance

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **RMSE** | 0.5072 | **1.1631** |
| **R²** | 0.9403 | **0.7138** |
| **MAE** | - | **0.8825** |

### Key Findings

 **Decent generalization**: Test R² of 0.71 indicates the model captures ~71% of variance  
 **Expected overfitting**: Train R² (0.94) vs Test R² (0.71) shows some overfitting  
 **Room for improvement**: RMSE of 1.16 suggests predictions can be off by ~1 log unit  
 **Practical utility**: MAE of 0.88 is reasonable for early-stage screening

### Model Interpretation

**What the metrics mean:**
- **RMSE = 1.16**: Average prediction error of ~1.16 log units
- **R² = 0.71**: Model explains 71% of solubility variance
- **MAE = 0.88**: Typical absolute error is 0.88 log units

**For context:**
- 1 log unit difference = 10× change in solubility
- RMSE of 1.16 ≈ predictions within 15× of actual value
- This is acceptable for prioritizing molecules in early drug discovery

---



##  Installation

### Requirements
- Python 3.12+
- 4GB+ RAM
- CUDA (optional, for GNN training on GPU)

### Core Dependencies

```bash
pip install rdkit pandas numpy scikit-learn matplotlib seaborn joblib
```

### For Graph Neural Networks (Optional)

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric
pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```



##  Model Details

### Random Forest Configuration

```python
RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Trees grown to maximum depth
    min_samples_split=2,   # Default split criterion
    min_samples_leaf=1,    # Default leaf size
    max_features='sqrt',   # Random feature subset
    random_state=42,       # Reproducibility
    n_jobs=-1             # Parallel processing
)
```

### Feature Importance

The Random Forest model learns which molecular substructures are most predictive of solubility. Top contributing fingerprint bits often correspond to:
- Hydrogen bond donors/acceptors
- Lipophilic groups
- Aromatic rings
- Polar functional groups

---

## Example Predictions

### Common Pharmaceutical Compounds

| Drug        | SMILES | Predicted log(S) | Interpretation |
|-------------|--------|------------------|----------------|
| **Ibuprofen** | `CC(C)Cc1ccc(cc1)C(C)C(O)=O` | -3.02 | Moderately Soluble |
| **Aspirin** | `CC(=O)Oc1ccccc1C(=O)O` | -3.00 | Moderately Soluble |
| **Caffeine** | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | -3.20 | Moderately Soluble |
| **Paracetamol** | `CC(=O)Nc1ccc(O)cc1` | -3.09 | Moderately Soluble |
| **Warfarin** | `CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O` | -3.06 | Moderately Soluble |

**Note**: These are model predictions for demonstration purposes. Always validate with experimental data for critical applications.

---

##  Learnings & Reflections

### What Worked Well

 **Data quality**: ESOL dataset is clean, well-curated, and widely used  
 **Morgan fingerprints**: Proven effective representation for molecular properties  
 **Random Forest**: Robust, interpretable, and requires minimal hyperparameter tuning  
 **Pipeline design**: Clear separation of featurization, training, and evaluation  
 **Reproducibility**: Fixed random seeds ensure consistent results

### Challenges Encountered

 **Overfitting**: Significant gap between train (0.94) and test R² (0.71)  
 **Limited data**: Only 1,128 molecules may not capture full chemical diversity  
 **Feature engineering**: Morgan fingerprints lose some 3D information  
 **GNN complexity**: Graph Neural Networks more complex to implement and tune  
 **Computational cost**: GNN training slower than Random Forest

### Key Takeaways

 **Domain knowledge matters**: Understanding chemistry helps interpret results  
 **Baselines are valuable**: Simple models (RF) often surprisingly effective  
 **Data > models**: Quality dataset more important than complex architectures  
 **Evaluation is critical**: Multiple metrics reveal different aspects of performance  
 **Iteration is key**: First attempt won't be perfect—that's okay!


## Performance Analysis

### Error Distribution

The model performs better on molecules with:
- **Moderate size** (10-30 heavy atoms)
- **Common functional groups** (well-represented in training)
- **Moderate solubility** (middle of distribution)

The model struggles with:
- **Extreme solubility** (very high or very low)
- **Novel scaffolds** (unlike training molecules)
- **Large molecules** (>40 heavy atoms)

### Residual Analysis

Residual plot shows:
- Generally centered around zero (good)
- Slight heteroscedasticity (variance increases with predicted value)
- Some outliers with errors >2 log units

---

##  References

### Key Papers
- Delaney, J. S. (2004). "ESOL: Estimating aqueous solubility directly from molecular structure." *Journal of Chemical Information and Computer Sciences*, 44(3), 1000-1005.
- Rogers, D., & Hahn, M. (2010). "Extended-connectivity fingerprints." *Journal of Chemical Information and Modeling*, 50(5), 742-754.
- Duvenaud, D. K., et al. (2015). "Convolutional networks on graphs for learning molecular fingerprints." *NeurIPS 2015*.

