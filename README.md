# Breast Cancer Survival Prediction

A deep learning project that predicts breast cancer survival outcomes using advanced neural network architectures with Fourier feature engineering. This project demonstrates the application of machine learning in medical data analysis for clinical decision support.

## 📋 Project Overview

This project implements a sophisticated neural network approach to predict breast cancer patient survival using clinical and pathological features. The model employs advanced techniques including Fourier feature transformation to capture complex non-linear relationships in medical data.

**Key Features:**
- **Dual Methodology**: Comparison between Fourier-enhanced and standard neural networks
- **Comprehensive Data Preprocessing**: Handles mixed categorical and numerical medical data
- **Advanced Feature Engineering**: Fourier series transformation for enhanced pattern recognition
- **Medical Classification**: Predicts patient survival status with high accuracy
- **Scalable Training**: Progressive training with different dataset sizes

## 🏥 Medical Application

### Clinical Significance
- **Patient Risk Assessment**: Helps clinicians evaluate survival probability
- **Treatment Planning**: Supports personalized treatment decision-making
- **Prognosis Prediction**: Provides data-driven survival insights
- **Healthcare Analytics**: Contributes to population health management

### Dataset Features
The model analyzes **16 critical clinical variables** from **4,024 breast cancer patients**:

#### Demographic Features
- Age, Race, Marital Status

#### Clinical Staging
- T Stage (Tumor size classification)
- N Stage (Lymph node involvement)  
- 6th Stage (Overall cancer stage)
- A Stage (Anatomical stage)

#### Pathological Characteristics
- Tumor differentiation and grade
- Tumor size (continuous measurement)
- Estrogen/Progesterone receptor status

#### Treatment Response
- Regional lymph nodes examined/positive
- Survival months
- **Target**: Status (Alive/Dead)

**Dataset Distribution:**
- **Total Patients**: 4,024
- **Survival Rate**: 84.7% (3,408 alive, 616 deceased)
- **Balanced Analysis**: Accounts for class imbalance in medical data

## 🔬 Technical Implementation

### Architecture Comparison

#### 1. Fourier-Enhanced Neural Network (`main.ipynb`)
```python
class SurvivalPredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, fourier_order=1):
        # Fourier feature transformation in forward pass
        fourier_features = torch.cat([torch.sin(orders * x), torch.cos(orders * x), x], dim=-1)
        # Multi-layer classification with softmax output
```

**Advanced Features:**
- **Fourier Series Expansion**: Captures periodic patterns in medical data
- **Multi-layer Architecture**: 64→32→16→2 neurons with ReLU activation
- **Classification Output**: Softmax for survival probability distribution
- **GPU Acceleration**: CUDA support for faster training

#### 2. Standard Neural Network (`withoutfourier.ipynb`)
```python
class SurvivalPredictionNN(nn.Module):
    # Traditional feedforward architecture
    # Regression approach for survival time prediction
```

**Baseline Features:**
- **Standard Architecture**: Traditional feedforward network
- **Regression Output**: Direct survival time prediction
- **Comparative Analysis**: Benchmarks Fourier enhancement effectiveness

### Data Processing Pipeline

#### 1. Preprocessing Strategy
```python
# Medical data handling
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

#### 2. Feature Engineering
- **Label Encoding**: A Stage (anatomical staging)
- **One-Hot Encoding**: Categorical medical variables
- **Standard Scaling**: Numerical clinical measurements
- **Missing Value Imputation**: Median/mode strategies for medical data

#### 3. Target Encoding
- **Binary Classification**: Alive (0) vs Dead (1)
- **Medical Relevance**: Clinically interpretable outcomes

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch pandas scikit-learn numpy tqdm
```

### CUDA Support (Optional)
For GPU acceleration:
```bash
# Check PyTorch CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Usage

#### 1. Fourier-Enhanced Model
```bash
jupyter notebook main.ipynb
```
- Run all cells sequentially
- Model trains with progressive dataset sizes (20%, 40%, 60%, 80%, 100%)
- Outputs classification accuracy for each training size

#### 2. Standard Baseline Model
```bash
jupyter notebook withoutfourier.ipynb
```
- Comparative baseline without Fourier features
- Traditional regression approach
- Performance benchmarking

### Example Training Output
```
Training with 20.0% of the data:
Validation Loss: 0.5234
Test Loss: 0.5156
Accuracy: 84.23%

Training with 40.0% of the data:
Validation Loss: 0.4891
Test Loss: 0.4776
Accuracy: 86.45%
...
```

## 📊 Model Architecture

### Fourier-Enhanced Network
```
Input Layer (15 clinical features)
    ↓ Fourier Transformation (sin/cos series)
Expanded Features (45 dimensions)
    ↓ ReLU Activation
Hidden Layer 1 (64 neurons)
    ↓ ReLU Activation  
Hidden Layer 2 (32 neurons)
    ↓ ReLU Activation
Hidden Layer 3 (16 neurons)
    ↓ Softmax Activation
Output Layer (2 classes: Alive/Dead)
```

### Training Configuration
- **Loss Function**: CrossEntropyLoss (classification)
- **Optimizer**: Adam (lr=0.0001)
- **Batch Size**: 32 patients
- **Epochs**: 50 (Fourier) / 30 (Standard)
- **Train/Test Split**: 75%/25%
- **Device**: GPU (CUDA) when available

## 🎯 Medical AI Innovation

### Advanced Techniques
1. **Fourier Feature Engineering**
   - Captures periodic patterns in patient data
   - Enhances model's ability to detect complex medical relationships
   - Novel application in survival prediction

2. **Progressive Training Analysis**
   - Evaluates model performance across different data sizes
   - Demonstrates learning curve and data efficiency
   - Critical for medical AI where data collection is expensive

3. **Medical Data Preprocessing**
   - Handles missing values common in clinical datasets
   - Proper encoding of categorical medical variables
   - Maintains clinical interpretability

### Clinical Applications
- **Risk Stratification**: Identify high-risk patients
- **Treatment Optimization**: Guide therapy selection
- **Resource Planning**: Hospital capacity and care coordination
- **Research Support**: Clinical trial design and patient selection

## 📁 Project Structure
```
Breast cancer survival prediction/
├── main.ipynb                    # Fourier-enhanced neural network
├── withoutfourier.ipynb         # Standard neural network baseline
├── Breast_Cancer.csv            # Clinical dataset (4,024 patients)
└── README.md                    # Project documentation
```

## 🛠️ Technical Skills Demonstrated

### Deep Learning & AI
- **PyTorch Implementation**: Custom neural network architectures
- **Feature Engineering**: Fourier series mathematical transformations
- **Medical AI**: Healthcare-specific machine learning applications
- **Model Comparison**: Systematic evaluation of different approaches

### Data Science & Analytics
- **Medical Data Processing**: Clinical dataset preprocessing
- **Statistical Analysis**: Survival analysis and classification metrics
- **Experimental Design**: Progressive training methodology
- **Performance Evaluation**: Accuracy assessment across data sizes

### Software Engineering
- **Modular Design**: Clean, reusable neural network classes
- **GPU Programming**: CUDA acceleration implementation
- **Jupyter Notebooks**: Interactive development and presentation
- **Documentation**: Clear, medical-domain-aware code documentation

### Healthcare Technology
- **Clinical Data Understanding**: Medical terminology and staging systems
- **Regulatory Awareness**: Healthcare AI considerations
- **Interpretable AI**: Clinically meaningful model outputs
- **Ethical AI**: Responsible medical AI development

## 🔮 Future Enhancements

### Model Improvements
- **Ensemble Methods**: Combine multiple neural architectures
- **Attention Mechanisms**: Focus on critical clinical features
- **Survival Analysis**: Time-to-event modeling with censored data
- **Explainable AI**: SHAP values for clinical interpretation

### Clinical Integration
- **Real-time Prediction API**: Hospital system integration
- **Decision Support Interface**: Clinician-friendly visualization
- **Multi-center Validation**: Cross-hospital model testing
- **Regulatory Compliance**: FDA pathway for medical devices

### Research Extensions
- **Multi-modal Data**: Integration of imaging and genomic data
- **Longitudinal Modeling**: Patient trajectory prediction
- **Personalized Medicine**: Treatment response prediction
- **Population Health**: Epidemiological pattern analysis

## 🌟 Business Impact

This project demonstrates practical applications in:
- **Healthcare Technology**: Medical AI and decision support systems
- **Pharmaceutical Research**: Drug development and clinical trials
- **Health Insurance**: Risk assessment and actuarial modeling
- **Public Health**: Population screening and prevention programs

---

*This project showcases the intersection of advanced machine learning, medical domain expertise, and clinical application development - demonstrating comprehensive skills in healthcare AI and medical data science.*