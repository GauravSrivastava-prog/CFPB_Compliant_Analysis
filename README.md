# CFPB Consumer Intelligence Platform

An end-to-end Data Science and Natural Language Processing (NLP) framework designed to analyze, visualize, and predict consumer dispute behaviors. This repository serves as a comprehensive academic assignment focusing on data engineering, statistical validation, and predictive modeling using consumer financial data.

---

## Dataset Description

This project utilizes the **Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database**.

The CFPB dataset is a collection of complaints about consumer financial products and services sent to companies for response. It is a highly valuable asset for academic analysis because it contains highly dimensional data, including:
- **Categorical Meta-Features**: Products (e.g., Mortgages, Credit Cards), Sub-products, Issues, and Corporate Entities.
- **Temporal Data**: Dates received and dates sent to companies, allowing for longitudinal operational analysis.
- **Unstructured Textual Data**: Consumer complaint narratives which provide raw, unfiltered consumer sentiment and semantic structures.
- **Target Variables**: Categorical indications of company responses, specific resolution delays, and historical markers of consumer dispute friction.

By leveraging this dataset, the assignment explores methodologies for transforming sparse, unstructured financial data into structured predictive architectures.

---

## The Core Components & Scientific Methodology

### 1. The Analytical Notebook
> **File:** `CFPB_Complaint_Analysis.ipynb`

This notebook executes the entire data lifecycle in a rigorously structured format. It utilizes isolated execution cells for methodological clarity and reproducibility:

*   **Exploratory Data Analysis (EDA)**: Structured graphical representations mapping timelines, categorical volume, and operational reaction delays.
*   **Statistical Reasoning Framework**:
    *   **Welch's T-Test**: Utilized to evaluate statistical differences between the means of complaint text length against binary dispute outcomes without assuming equal population variance.
    *   **Chi-Square Contingency Test**: Calculates the goodness of fit between categorical variables (Financial Product Types vs. Dispute Outcomes) to determine statistical independence.
    *   **ANOVA (Analysis of Variance)**: Assesses the variance of continuous delay times across multiple corporate entities to scientifically validate systematic delays.
*   **Time Series Analytics**: Implementation of **ARIMA** (Auto-Regressive Integrated Moving Average) modeling to map, smooth out noise, and forecast complaint trajectories across chronological periods based on historical stationarity.
*   **Unsupervised NLP Modeling**: 
    *   **Latent Dirichlet Allocation (LDA)**: A generative statistical model that allows sets of observations to be explained by unobserved groups, used here for hierarchical topic extraction from narratives.
    *   **Principal Component Analysis (PCA)**: An orthogonal linear transformation applied to TF-IDF matrices to project highly dimensional semantic arrays into 2D evaluative space.
*   **Supervised Machine Learning**: Comparative evaluation of **Linear Regression**, **Logistic Regression**, and **Random Forest Classifiers**. Model performance is quantitatively assessed using Precision/Recall matrices, ROC (Receiver Operating Characteristic) curves, and Confusion Matrices.

### 2. Systemic Analysis Terminal (Dashboard)
> **Execution Path:** `dashboard/streamlit/app.py`

A multi-page Web Interface designed to visually demonstrate the statistical findings of the notebook via interactive environments.

*   **Multi-Page Architecture**: The dashboard routing separates the `Industrial Analytics` descriptive statistics hub from the predictive modeling environments.
*   **Dynamic Visualizations**: Integration of dimensional `Plotly` components (Treemaps, Violin distribution plots mapping probability densities, Pearson Correlation Heatmaps mapping -1 to 1 systemic correlation).
*   **Live Hypothesis Engine**: A computational interface that dynamically executes `scipy.stats` computations in real-time acting on the loaded DataFrame frame.

### 3. The NLP Prediction Engine

Located within the secondary routing page of the Streamlit Dashboard, this module represents the practical deployment of semantic modeling methodologies.

Rather than predicting risk based solely on metadata, the backend logic executes a live **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization Matrix**. TF-IDF evaluates terminology by rewarding specificity and penalizing commonality (down-weighting semantic noise like 'and', 'the').

1. The exact textual features are extracted and vectorized from injected test strings.
2. The engine concatenates the TF-IDF semantic array with a live **TextBlob Polarity Matrix** (assigning continuous values from -1.0 to 1.0).
3. The normalized array is evaluated by the non-linear boundaries of the **Random Forest Classifier**.
4. The system outputs a formalized probability scale dictating the statistical likelihood of a prolonged consumer dispute.

#### Sample NLP Test Cases
To test the semantic capabilities of the Prediction Engine within the Streamlit dashboard, try the following structural narratives:

*   **Test Case A (Resolution Probable - Low Risk):**
    > *"Thank you for the update. The bank was mostly polite but I did experience a minor delay receiving my replacement card. The issue is mostly resolved now."*
    *Expected Behavior*: Model flags low negative sentiment and standard vocabulary, predicting low dispute latency.

*   **Test Case B (High Dispute Latency - High Risk):**
    > *"This is a complete scam and absolutely terrible service! They committed fraud, stole my money from my private account, and I will be contacting a lawyer to sue them immediately. Completely unacceptable and illegal behavior."*
    *Expected Behavior*: Model vectorizer intercepts severe threat vectors ('scam', 'fraud', 'steal', 'lawyer') combined with extreme negative polarity to predict high probability of an active dispute.

---

## Technical Stack 
- **Core Languages**: Python 3.x
- **Data Engineering**: Pandas, Numpy
- **Statistical Modeling**: Scikit-Learn, Statsmodels, Scipy
- **Natural Language Processing**: NLTK, TextBlob, WordCloud
- **Visualization**: Plotly Graph Objects, Seaborn, Matplotlib
- **Web Deployment**: Streamlit

## Setup & Execution

**1. Clone the project and provision the environment via Requirements:**
```bash
pip install -r requirements.txt
```

**2. Explore the Analytical Notebook:**
Execute the cells within `CFPB_Complaint_Analysis.ipynb` sequentially using a valid Jupyter environment.

**3. Launch the Analytical Dashboard:**
Navigate to the root directory and initiate the local Streamlit server logic (Noting the updated directory structure):
```bash
streamlit run dashboard/streamlit/app.py
```
*The Streamlit Application will execute locally on `http://localhost:8501`.*
