# Air Quality Forecasting with RNN/LSTM Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)]()

## Project Overview

This project focuses on applying advanced Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models to forecast PM2.5 air pollution levels in Beijing. The goal is to achieve accurate predictions with a Root Mean Squared Error (RMSE) below 4000 using historical air quality and weather data.

**🎯 Objective**: Develop a robust deep learning system for accurate air quality forecasting to support public health decision-making and urban planning.

## Problem Statement

Air pollution, particularly PM2.5, is a critical global issue impacting public health and urban planning. PM2.5 particles are small enough to penetrate deep into the lungs and bloodstream, causing serious health problems. By accurately predicting PM2.5 concentrations, governments and communities can:

- Issue timely health warnings to vulnerable populations
- Implement traffic restrictions during high pollution periods
- Plan outdoor activities and events safely
- Develop targeted environmental policies

## 📊 Dataset Description

The dataset contains comprehensive historical air quality and weather data for Beijing:

**Features**:

- `pm2.5`: PM2.5 concentration (µg/m³) - **TARGET VARIABLE**
- `DEWP`: Dew point temperature (°C)
- `TEMP`: Temperature (°C)
- `PRES`: Atmospheric pressure (hPa)
- `Iws`: Cumulated wind speed (m/s)
- `Is`: Cumulated precipitation (mm)
- `Ir`: Cumulated rainfall (mm)
- `cbwd_NW/SE/cv`: Wind direction indicators (binary)
- `datetime`: Timestamp for temporal analysis

**Data Characteristics**:

- Hourly measurements over multiple years
- Rich temporal patterns (diurnal, seasonal)
- Complex relationships between meteorological and pollution variables

## 🏗️ Project Structure

```
Air quality/
├── data/                          # Raw and processed data files
│   ├── train.csv                 # Training dataset from Kaggle
│   ├── test.csv                  # Test dataset for predictions
│   └── sample_submission.csv     # Submission format template
├── notebooks/                     # Jupyter notebooks for analysis
│   └── air_quality_forecasting_starter_code.ipynb  # Main analysis notebook
├── scripts/                       # Python utility modules
│   ├── data_preprocessing.py     # Advanced preprocessing pipeline
│   ├── model_architectures.py    # RNN/LSTM model definitions
│   └── experiment_tracker.py     # Experiment management system
├── models/                        # Saved model files (.h5)
├── outputs/                       # Results, plots, and submissions
│   ├── exploratory_data_analysis.png
│   ├── time_series_analysis.png
│   ├── experiment_analysis.png
│   ├── final_model_evaluation.png
│   ├── test_predictions_analysis.png
│   ├── experiment_results.csv
│   └── air_quality_submission_exp_XX.csv
└── requirements.txt               # Python dependencies
```

## 🔬 Methodology

### 1. **Comprehensive Data Exploration**

- Statistical analysis with summary statistics and distributions
- Time series visualization revealing diurnal and seasonal patterns
- Correlation analysis between weather variables and air quality
- Missing value analysis and imputation strategies
- Pollution level categorization according to WHO guidelines

### 2. **Advanced Data Preprocessing**

- **Missing Value Handling**: Multi-strategy approach (forward/backward fill, interpolation)
- **Feature Engineering**:
  - Lag features (1h, 2h, 3h, 6h, 12h, 24h)
  - Rolling statistics (mean, std, min, max) over multiple windows
  - Temporal features with cyclical encoding
  - Weather interaction features (humidity, wind chill, pressure-temperature)
- **Normalization**: StandardScaler for features, MinMaxScaler for target
- **Sequence Creation**: Sliding window approach with 24-hour lookback

### 3. **Model Architectures**

- **Simple LSTM**: Baseline single-layer architecture
- **Deep LSTM**: Multi-layer networks with dropout regularization
- **Bidirectional LSTM**: Enhanced temporal context capture
- **GRU Models**: Alternative recurrent architecture
- **CNN-LSTM Hybrid**: Feature extraction + temporal modeling
- **Attention LSTM**: Simplified attention mechanism

### 4. **Systematic Experimentation**

- **18 comprehensive experiments** with varied:
  - Model architectures (6 different types)
  - Learning rates (0.0003 to 0.005)
  - Batch sizes (8 to 128)
  - Optimizers (Adam, RMSprop, SGD)
  - Loss functions (MSE, MAE, RMSE, Huber)
  - Regularization parameters
- Proper train/validation split with temporal ordering
- Early stopping and learning rate scheduling

### 5. **Comprehensive Evaluation**

- **Primary Metric**: RMSE (Root Mean Squared Error)
- **Additional Metrics**: MAE, R², MAPE
- **Performance Analysis**: By pollution level, time of day, season
- **Model Comparison**: Architecture effectiveness, training efficiency
- **Error Analysis**: Residual plots, overfitting assessment

## 🎯 Key Features

### ✅ Data Analysis Excellence

- **13 comprehensive visualizations** showing temporal patterns, correlations, and distributions
- **Statistical insights** with pollution level analysis and WHO guideline comparisons
- **Time series decomposition** revealing diurnal and seasonal patterns

### ✅ Advanced Feature Engineering

- **50+ engineered features** from original 11 variables
- **Temporal encoding** capturing cyclical patterns (hour, day, month)
- **Weather interactions** based on atmospheric physics
- **Lag and rolling features** for temporal dependencies

### ✅ Robust Model Development

- **6 different architectures** systematically compared
- **Advanced regularization** (dropout, early stopping, L2)
- **Proper sequence modeling** with 24-hour context windows
- **RNN challenge mitigation** (vanishing gradients, overfitting)

### ✅ Systematic Experimentation

- **18 experiments** with comprehensive parameter variations
- **Automated tracking** of all hyperparameters and results
- **Statistical comparison** of architectural effectiveness
- **Best model selection** based on validation performance

## 📈 Results Summary

### Model Performance

- **Best Architecture**: [To be determined after running experiments]
- **Validation RMSE**: [Target: < 4000 µg/m³]
- **Training Efficiency**: Optimized with early stopping and LR scheduling
- **Generalization**: Monitored through train/validation RMSE ratio

### Key Insights

- **Temporal Patterns**: Strong diurnal cycles with peak pollution during rush hours
- **Seasonal Variations**: Higher pollution in winter months due to heating
- **Weather Impact**: Significant correlations with temperature, pressure, and wind
- **Model Comparison**: [Architecture effectiveness to be documented]

### Practical Applications

- Real-time air quality forecasting system
- Public health early warning system
- Environmental policy impact assessment
- Urban planning decision support

## 🔧 Technical Requirements

```bash
Python >= 3.8
TensorFlow >= 2.13.0
pandas >= 1.5.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
jupyter >= 1.0.0
```

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone [repository-url]
cd "Air quality"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Data

1. Join the Kaggle competition: [Air Quality Forecasting]
2. Download the following files:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`
3. Place files in the `data/` directory

### 4. Run Analysis

```bash
jupyter notebook notebooks/air_quality_forecasting_starter_code.ipynb
```

## 📊 Usage Instructions

### Quick Start

1. **Open the main notebook**: `air_quality_forecasting_starter_code.ipynb`
2. **Run all cells sequentially** - the notebook is designed for linear execution
3. **Monitor progress** - comprehensive logging shows experiment progress
4. **Review results** - generated visualizations and tables in `outputs/` folder
5. **Kaggle submission** - ready-to-submit CSV file generated automatically

### Customization

- **Modify experiments**: Edit the experiment configurations in the notebook
- **Add new models**: Use `scripts/model_architectures.py` as template
- **Change preprocessing**: Customize `scripts/data_preprocessing.py`
- **Experiment tracking**: Leverage `scripts/experiment_tracker.py`

## 📋 Expected Outputs

### Generated Files

- `experiment_results.csv`: Comprehensive experiment comparison table
- `best_model_exp_XX.h5`: Best performing model weights
- `air_quality_submission_exp_XX.csv`: Kaggle submission file
- Multiple visualization PNG files showing analysis results

### Key Visualizations

1. **Exploratory Data Analysis**: Feature distributions and correlations
2. **Time Series Analysis**: Temporal patterns and seasonality
3. **Experiment Comparison**: Model performance across configurations
4. **Final Evaluation**: Prediction accuracy and error analysis
5. **Test Predictions**: Submission data analysis and validation

## 🧠 RNN Challenges & Solutions

### Common Challenges

- **Vanishing Gradients**: Gradients become very small in deep networks
- **Exploding Gradients**: Gradients become very large, causing instability
- **Long-term Dependencies**: Difficulty learning relationships across long sequences
- **Computational Complexity**: Training can be slow for long sequences

### Implemented Solutions

- **LSTM/GRU Cells**: Designed to handle vanishing gradients better than vanilla RNNs
- **Gradient Clipping**: Implicit in Adam optimizer to handle exploding gradients
- **Dropout Regularization**: Applied to prevent overfitting
- **Early Stopping**: Prevents overtraining and reduces computational cost
- **Learning Rate Scheduling**: ReduceLROnPlateau to adapt learning rate
- **Proper Sequence Length**: Balanced 24-hour context window for efficiency
- **Bidirectional Processing**: Captures both forward and backward dependencies

## 🎓 Educational Value

### Learning Outcomes

- **Time Series Forecasting**: Master RNN/LSTM applications to sequential data
- **Deep Learning Best Practices**: Systematic experimentation and evaluation
- **Real-world Problem Solving**: Environmental data analysis and preprocessing
- **Model Comparison**: Understanding architectural trade-offs and performance
- **Research Methodology**: Comprehensive analysis and scientific reporting

### Skill Development

- Advanced TensorFlow/Keras programming
- Time series data preprocessing and feature engineering
- Systematic machine learning experimentation
- Scientific visualization and result interpretation
- Environmental data science and domain knowledge

## 📚 References

[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. _Neural Computation_, 9(8), 1735-1780.

[2] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. _EMNLP_.

[3] World Health Organization. (2021). WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide.

[4] Zhang, L., Liu, P., Zhao, L., Wang, G., Zhang, W., & Liu, J. (2018). Deep learning for air quality prediction: A review. _Atmospheric Environment_, 195, 148-157.

[5] TensorFlow Documentation. (2023). Time series forecasting. https://www.tensorflow.org/tutorials/structured_data/time_series

[6] Kaggle Competition: Air Quality Forecasting Challenge. https://www.kaggle.com/competitions/[competition-name]

## 👥 Contributing

This project is part of the ALU Machine Learning Techniques I course. For educational purposes:

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** improvements or extensions
4. **Add** comprehensive documentation
5. **Submit** a pull request with detailed description

## 📄 License

This project is developed for educational purposes as part of the African Leadership University (ALU) Machine Learning Techniques I course. The code is available for learning and academic use.

## 🏆 Acknowledgments

- **ALU Faculty**: For providing comprehensive ML education and project guidance
- **Kaggle Community**: For hosting the air quality forecasting competition
- **TensorFlow Team**: For excellent deep learning framework and documentation
- **Environmental Scientists**: For domain expertise in air quality modeling

---
