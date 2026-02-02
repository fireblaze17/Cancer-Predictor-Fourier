# Breast Cancer Survival Prediction

A deep learning project that predicts breast cancer survival outcomes using advanced neural network architectures with Fourier feature engineering. This project demonstrates the application of machine learning techniques in a challenging healthcare domain.

## Project Overview

This project implements a sophisticated neural network approach to predict breast cancer patient survival using clinical and pathological features. The model employs advanced techniques including Fourier feature engineering and GPU acceleration for faster computation.

### Key Features:
- **Dual Methodology**: Comparison between Fourier-enhanced and standard neural networks.
- **Comprehensive Data Preprocessing**: Handles mixed categorical and numerical medical data.
- **Advanced Feature Engineering**: Fourier series transformation for enhanced pattern recognition.
- **Medical Classification**: Predicts patient survival status with high accuracy.
- **Scalable Training**: Progressive training with different dataset sizes.

### Dataset

The model analyzes **16 critical clinical variables** from **4,024 breast cancer patients**:
1. **Demographic Features**: Age, Race, Marital Status.
2. **Clinical Staging**: Tumor size, lymph node involvement, overall staging.
3. **Pathological Features**: Tumor differentiation and grade, receptor status.
4. **Treatment Response**: Lymph nodes examined, Survival months, Status (Target: Alive/Dead).


## Technical Implementation

### Fourier-Enhanced Neural Network
#### Architecture:
- Uses Fourier Series Expansion to enhance medical features.
- Neural architecture: 64→32→16→2 neurons with ReLU activation and Softmax output.

#### Baseline Model
- Traditional feedforward neural network for direct survival time prediction.
- Comparative analysis demonstrates Fourier enhancements' effectiveness.

### Data Processing Pipeline
- **Preprocessing**: Numerical scaling, categorical encoding, outlier management.
- **Feature Engineering**: Fourier transformations, imputation using median/mode values.
- **Medical Insights**: Focus on clinically interpretable outcomes.

## Getting Started

### Prerequisites
```bash
pip install torch pandas scikit-learn numpy tqdm
```

### Usage

#### Train Fourier-Enhanced Model
```bash
jupyter notebook main.ipynb
```
Use Fourier-transformed features to train a survival prediction neural network.

#### Baseline Model
```bash
jupyter notebook withoutfourier.ipynb
```
Run the traditional neural network to benchmark Fourier enhancements.

### Output Example
```
Training with Fourier Model:
Accuracy: 86.45%
Without Fourier:
Accuracy: 82.68%
```

## Lessons Learned

Throughout this project, I acquired technical and domain-specific knowledge:

### Technical:
1. The power of Fourier feature engineering in complex domains like healthcare.
2. Optimizing neural networks for small-sized, imbalanced datasets.
3. Incorporating GPU-based acceleration using PyTorch.

### Healthcare Understanding:
1. Learned the intricacies of clinical data preprocessing and its challenges (e.g., missing data)
2. Studied the role of explainable AI for healthcare applications.
3. Developed a deep respect for ensuring ethical and interpretable AI solutions in sensitive domains.

*This project exemplifies advanced applications in machine learning, tailored for medical challenges, while maintaining reproducibility and clear documentation.*