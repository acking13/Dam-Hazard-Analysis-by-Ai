<div align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/Project-Dam%20Hazard%20Prediction-blue?style=for-the-badge" alt="Project Badge">
  </a>
  <h1 align="center">🌊 Predictive Pipeline for Dam Hazard Analysis 🌊</h1>
  <p align="center">
    A sophisticated deep learning pipeline to classify and predict dam hazard incidents from historical data.
    <br />
    <a href="#-about-the-project"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#-results-and-discussion">View Results</a>
    ·
    <a href="https://github.com/YOUR_USERNAME/YOUR_REPO/issues">Report Bug</a>
    ·
    <a href="https://github.com/YOUR_USERNAME/YOUR_REPO/issues">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#-about-the-project">About The Project</a></li>
    <li><a href="#-methodology-and-key-features">Methodology</a></li>
    <li><a href="#-automated-workflow-and-outputs">Workflow & Outputs</a></li>
    <li><a href="#-results-and-discussion">Results & Discussion</a></li>
    <li><a href="#-appendix-summary-of-model-performance-metrics">Appendix</a></li>
  </ol>
</details>

---

## 🧐 About The Project

Ensuring the safety and integrity of dams is a critical task in civil infrastructure management. Proactive risk assessment, which involves anticipating potential failure modes and their consequences, is paramount to preventing catastrophic events. This project introduces a sophisticated machine learning pipeline designed to transform raw historical data into **actionable intelligence** for risk managers and engineers.

The core of this project is a Python script that automates the end-to-end process of data preparation, model training, evaluation, and reporting. It leverages a robust **Deep Neural Network (DNN)** to build multiple classification models, each tailored to predict a specific aspect of a potential dam failure incident.

---

## 🛠️ Methodology and Key Features

The pipeline is engineered with modern machine learning best practices to ensure reliability, accuracy, and robustness.

🧠 **Advanced Neural Network Architecture**
> A DNN built with TensorFlow and Keras, featuring multiple dense layers with ReLU activation, Batch Normalization, Dropout, and L2 regularization to prevent overfitting.

⚙️ **Comprehensive Preprocessing**
> A scikit-learn pipeline automatically handles numerical features (`StandardScaler`) and categorical features (`OneHotEncoder`), making the system highly adaptable.

⚖️ **Imbalanced Data Handling**
> Integration of **SMOTE** (Synthetic Minority Over-sampling Technique) to intelligently resample training data, ensuring the model effectively learns to recognize minority classes.

🎯 **Multi-Target Classification**
> The system iteratively trains and evaluates separate, specialized models for a range of target variables, such as `incident_type`, `incident_mechanism`, and `response`.

⏱️ **Optimized Training Process**
> Training is enhanced with an Adam optimizer, `EarlyStopping` to find the optimal number of epochs, and `ReduceLROnPlateau` to dynamically adjust the learning rate for efficient convergence.

---

## 🚀 Automated Workflow and Outputs

The script executes a complete, automated workflow for each target variable:

1.  **Load & Clean**: The dataset is loaded and prepared for processing.
2.  **Preprocess & Split**: Features are processed, and data is split into training/testing sets.
3.  **Apply SMOTE**: The training set is balanced to correct for class imbalances.
4.  **Build, Compile & Train**: The deep neural network is constructed and trained.
5.  **Evaluate**: The trained model is evaluated on unseen test data.

Upon completion, the pipeline automatically generates a suite of outputs for comprehensive analysis:

* 💾 **Trained Model Files (`.h5`)**: Saved Keras models ready for deployment.
* 📊 **Performance Metrics Summary (`.xlsx`)**: A consolidated Excel report quantifying each model's Accuracy, Precision, Recall, and F1-Score.
* 📈 **Visual Confusion Matrices (`.svg`)**: High-quality plots for a clear visual breakdown of predictive accuracy.
* 📋 **Detailed Prediction Reports (`.xlsx`)**: Granular reports comparing actual vs. predicted outcomes for in-depth error analysis.

---

## 📊 Results and Discussion

The performance of the models varied significantly across prediction tasks, revealing which aspects of dam incidents are most predictable with the current methodology.


### 🏆 High-Performing Models
The models trained to predict **`response`** and **`incident_type`** demonstrated the highest efficacy. The `response` model achieved an exceptional **Accuracy of 0.97** and **F1-Score of 0.95**, indicating that operational responses follow clear, learnable patterns. The `incident_type` model performed strongly with an **Accuracy and F1-Score of ~0.80**.

<details>
  <summary>Click to see the Confusion Matrix for the <code>incident_type</code> model</summary>
  <br>
  <p align="center">
    <img src="https://placehold.co/600x400/f0f0f0/333?text=Figure+1:+Confusion+Matrix\nfor+incident_type" alt="Confusion Matrix for incident_type">
    <br>
    <em><b>Figure 1:</b> Confusion Matrix for <code>incident_type</code> Predictions. The model was highly effective at identifying "Overtopping" but showed some difficulty distinguishing "Structural Failure" and "Piping."</em>
  </p>
</details>
<br>

### 🤔 Moderately-Performing Models
The models for **`fatalities_number`** (Accuracy 0.70, F1-Score 0.69) and **`eap_enacted_y_n_due_to_incident`** (Accuracy 0.55, F1-Score 0.59) fall into a moderate performance category. They show some predictive power but require further refinement to be reliable for critical decision-making.

### 📉 Poorly-Performing Models & The Precision-Recall Imbalance
A key finding was the poor performance on rare events like `incident_mechanism_1`, `incident_mechanism_3`, and `other_infrastructure_impacts`. These models exhibited a classic imbalance issue: **deceptively high Precision but extremely low Recall and Accuracy**. For example, `incident_mechanism_3` had a Precision of 0.93 but a dismal Recall of 0.24. This indicates the model learned to "play it safe" by only predicting the majority class, making it unusable for detecting rare but critical failure events.

<div align="center">
  <img src="https://placehold.co/800x400/f0f0f0/333?text=Figure+2:+Comparison+of+Performance+Metrics\nAcross+All+Trained+Models" alt="Chart of Model Performance Metrics">
  <p><em><b>Figure 2:</b> Comparison of Performance Metrics Across All Trained Models.</em></p>
</div>

---

## 📋 Appendix: Summary of Model Performance Metrics

The table below provides a comprehensive summary of the performance metrics for all nine classification models. Precision, Recall, and F1-Score are weighted averages to account for class imbalance.

| Model Output                      | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :-------------------------------- | :------: | :------------------: | :---------------: | :-----------------: |
| **incident_type** |  0.796   |        0.796         |       0.796       |        0.796        |
| **incident_mechanism_1** |  0.130   |        0.241         |       0.130       |        0.110        |
| **incident_mechanism_2** |  0.101   |        0.730         |       0.101       |        0.152        |
| **incident_mechanism_3** |  0.235   |        0.929         |       0.235       |        0.371        |
| **eap_enacted_y_n_due_to_incident** |  0.554   |        0.658         |       0.554       |        0.587        |
| **fatalities_number** |  0.699   |        0.680         |       0.699       |        0.688        |
| **other_infrastructure_impacts** |  0.213   |        0.861         |       0.213       |        0.297        |
| **response** |  0.969   |        0.940         |       0.969       |        0.954        |
| **incident_report_produced** |  0.342   |        0.672         |       0.342       |        0.405        |

*Table 1: Consolidated Performance Metrics for All Predictive Models.*

# Predictive Modeling of Dam Incident Consequences: A Regression Analysis

This project explores the use of machine learning regression models to predict the numerical consequences of dam incidents. Using a dataset of historical incidents, various models, including **Random Forest** and **LightGBM**, were trained and evaluated to forecast key outcomes such as the volume of water released, dam height, and the number of people affected.

---

## 🎯 Project Goal

The primary objective was to determine the feasibility of accurately predicting the magnitude of dam failure consequences. The analysis aimed to identify which specific outcomes are predictable and which models provide the most reliable forecasts, ultimately providing insights for risk management and emergency preparedness.

---

## 🛠️ Methodology

The project followed a systematic machine learning workflow:

* **Multi-Target Regression:** Separate, specialized models were trained to predict each of the eight target variables (e.g., `dam_height`, `volume_released_at_failure_ac_ft`, etc.).
* **Model Comparison:** A suite of regression models was evaluated, with a focus on powerful ensemble techniques like Tuned Random Forest and LightGBM.
* **Performance Evaluation:** Model success was measured using standard regression metrics, primarily the **R² Score (Coefficient of Determination)** to assess predictive power and **Mean Absolute Error (MAE)** to quantify error magnitudes.

---

## 📊 Key Findings & Results

The analysis revealed that predicting the exact numerical consequences of dam incidents is highly challenging with the current feature set. The model performances varied significantly across the different targets.

> **Overall Conclusion:** While some physical outcomes show a weak predictive signal, consequences related to human impact and incident timing could not be reliably predicted.

### High-Performing Models 🤔
* The **most predictable outcome** was `volume_released_at_failure_ac_ft`.
* The best model, a **Tuned Random Forest (Model 8)**, achieved an **R² score of 0.351**, explaining about 35% of the variance in the data.

### Poorly-Performing Models 📉
* Physical attributes like `dam_height` (R² = 0.249) and `surface_area_acres` (R² = 0.237) showed very limited predictability.
* The models for these targets captured some patterns but left the majority of the variance unexplained.

### Unpredictable Outcomes 🚫
* Targets related to human impact (`number_of_people_evacuated`, `habitable_structures_evacuated`) and timing (`incident_duration`) were found to be **effectively unpredictable**.
* The best models for these variables had R² scores near zero (or even negative), meaning their predictions were no better than simply guessing the average value.

---

## 📈 Performance Summary

The chart below visualizes the best R² score achieved for each predictive target, highlighting the significant disparity in predictability across the outcomes.



<details>
<summary>Click to view the Python code used to generate this chart</summary>

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define the data based on the best model for each target
data = {
    'Target': [
        'dam_height', 'max_storage_ac_ft', 'surface_area_acres', 
        'people_evacuated', 'habitable_structures_evacuated', 
        'habitable_structures_flooded', 'volume_released_at_failure_ac_ft', 
        'incident_duration'
    ],
    'R2_Score': [0.249, 0.215, 0.237, 0.096, 0.102, 0.295, 0.351, 0.022]
}

# 2. Create a pandas DataFrame and sort by R² score
df_results = pd.DataFrame(data)
df_sorted = df_results.sort_values('R2_Score', ascending=False)

# 3. Create the plot using seaborn
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))

bars = sns.barplot(x='R2_Score', y='Target', data=df_sorted, palette='viridis', ax=ax)

# 4. Add titles, labels, and data values
ax.set_title('Comparison of Best R² Score by Predictive Target', fontsize=16, pad=20)
ax.set_xlabel('R² Score (Coefficient of Determination)', fontsize=12)
ax.set_ylabel('Predictive Target', fontsize=12)
ax.set_xlim(0, 0.4)

for bar in bars.patches:
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.3f}', va='center', fontsize=10)

# 5. Show the plot
sns.despine(left=True, bottom=True)
plt.tight_layout()
# plt.savefig('r2_score_comparison.png', dpi=300) # Uncomment to save
plt.show()
