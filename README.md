# Lifestyle Choices and Life Expectancy Analysis

A comprehensive data visualization project that analyzes how lifestyle choices impact life expectancy. This project generates realistic data based on research studies and WHO statistics, creates interactive visualizations, and provides personalized insights.

## 🚀 Features

- **Data Generation**: Creates comprehensive lifestyle combinations with realistic impact values
- **Interactive Visualizations**: Plotly-based interactive charts and graphs
- **Static Visualizations**: High-quality PNG images for presentations and social media
- **Personalized Analysis**: Streamlit dashboard for individual lifestyle assessment
- **Jupyter Notebooks**: Exploratory data analysis and insights
- **Production-Ready Code**: Comprehensive error handling, logging, and validation

## 📊 Project Structure

```
├── config.py                 # Configuration management
├── main.py                   # Main execution script
├── data_generator.py         # Data generation and processing
├── visualizer.py             # Visualization creation
├── interactive_dashboard.py  # Streamlit dashboard
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── logger.py            # Logging utilities
│   ├── validators.py        # Data and input validation
│   └── exceptions.py        # Custom exceptions
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_data_generator.py
│   └── test_validators.py
├── data/                     # Generated data files
├── exports/                  # Visualization outputs
├── notebooks/                # Jupyter notebooks
├── logs/                     # Application logs
└── requirements.txt          # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd lifestyle-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the project**
   ```bash
   python main.py
   ```

## 🎯 Usage

### Basic Usage

Run the complete project to generate data and visualizations:

```bash
python main.py
```

This will:
- Generate lifestyle data with 163,840 combinations
- Create interactive HTML visualizations
- Generate PNG images for presentations and social media
- Create a Jupyter notebook for analysis

### Interactive Dashboard

Launch the Streamlit dashboard for personalized analysis:

```bash
streamlit run interactive_dashboard.py
```

### Individual Components

**Generate data only:**
```bash
python data_generator.py
```

**Create visualizations only:**
```bash
python visualizer.py
```

**Run tests:**
```bash
pytest tests/
```

## ⚙️ Configuration

The project uses environment variables for configuration. Create a `.env` file or set environment variables:

```bash
# Application settings
DEBUG=false
HOST=localhost
PORT=8000

# Data settings
DATA_PATH=data
SAMPLE_SIZE=1000
RANDOM_STATE=42

# Visualization settings
OUTPUT_PATH=exports
DPI=300
FIGURE_SIZE=(16,12)

# Logging settings
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_MAX_SIZE=10485760
LOG_BACKUP_COUNT=5
```

## 📈 Data Structure

The generated dataset includes:

- **Age Groups**: 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69
- **Lifestyle Factors**:
  - Smoking: never, former, current_light, current_heavy
  - Exercise: sedentary, light, moderate, vigorous
  - Diet: poor, average, good, excellent
  - Alcohol: none, light, moderate, heavy
  - Sleep: poor, average, good, excellent
  - Stress: high, moderate, low, minimal
  - Social Connections: isolated, limited, moderate, strong

- **Calculated Metrics**:
  - Lifestyle Impact: Total years lost/gained
  - Adjusted Life Expectancy: Base expectancy + impact
  - Risk Score: 0-100 scale (higher = more risk)
  - Years Lost/Gained: Individual factor impacts

## 🔧 Development

### Code Quality

The project follows production-ready practices:

- **Type Hints**: Full type annotation throughout
- **Error Handling**: Comprehensive exception handling
- **Logging**: Structured logging with different levels
- **Validation**: Input and data validation
- **Testing**: Unit tests for all modules
- **Documentation**: Comprehensive docstrings

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_data_generator.py

# Run with verbose output
pytest -v
```

### Code Style

The project follows PEP 8 style guidelines. Use a formatter like `black`:

```bash
pip install black
black .
```

## 📊 Visualizations

### Interactive HTML
- Scatter plots showing lifestyle impact vs life expectancy
- Bar charts comparing factor impacts
- Risk score distribution histograms
- Age group comparison box plots

### Static PNG Images
- **Social Media**: Clean, shareable visualizations
- **Presentation**: Annotated charts for presentations

### Streamlit Dashboard
- Personalized lifestyle assessment
- Interactive factor selection
- Real-time impact calculation
- Improvement recommendations

## 🧪 Testing

The project includes comprehensive tests:

- **Unit Tests**: Individual function testing
- **Integration Tests**: Component interaction testing
- **Validation Tests**: Data and input validation
- **Error Handling Tests**: Exception scenario testing

Run tests with:
```bash
pytest tests/ -v
```

## 📝 Logging

The application uses structured logging:

- **Console Output**: INFO level and above
- **File Logging**: DEBUG level and above
- **Rotating Logs**: Automatic log rotation
- **Structured Format**: Timestamp, logger, level, message

Log files are stored in the `logs/` directory.

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production Deployment
1. Set production environment variables
2. Configure logging levels
3. Set up log rotation
4. Monitor application health

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data based on WHO statistics and research studies
- Visualization inspired by FlowingData
- Built with Python, Plotly, Matplotlib, and Streamlit

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the logs in the `logs/` directory
- Review the test cases for usage examples

## 🔄 Version History

- **v1.0.0**: Initial release with basic functionality
- **v1.1.0**: Added production-ready features
- **v1.2.0**: Enhanced error handling and logging
- **v1.3.0**: Added comprehensive testing suite