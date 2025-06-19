# Enhancing Sentiment Analysis for Low-Resource African Languages: Xitsonga, Sesotho, and Setswana

**Project by: Selloane Chalale, Khathophiwe Nemutudi, and Mulisa Thovhale (Group 17)**
**Date of Submission: 19 June 2025**

##  Project Overview

This project addresses the critical need for effective sentiment analysis tools in underrepresented African languages. We focus on **Xitsonga, Sesotho, and Setswana**, aiming to:
1.  Establish robust baseline performance using traditional machine learning models.
2.  Explore the efficacy of transfer learning with the Africa-centric **AfriBERTa** pre-trained model.
3.  Compare different fine-tuning strategies: **Full Fine-tuning** vs. parameter-efficient **LoRA Fine-tuning**.
4.  Investigate the impact of **data augmentation** (back-translation) on model accuracy.
5.  Employ model **explainability methods (SHAP)** to understand model predictions, errors, and potential biases.

This repository contains the Google Colab notebook demonstrating our experimental pipeline, from loading pre-cleaned datasets through model training, evaluation, and explainability analysis.

## Files and Data Structure

*   **Datasets:**
    *   The cleaned datasets (`.csv` format) for Xitsonga, Sesotho, and Setswana are expected to be located in a Google Drive folder accessible by the Colab notebook. The primary path used in the notebook is `/content/drive/MyDrive/NLPS1/Cleaned/`.
    *   **Xitsonga:** Derived from AfriSenti, with pre-defined `train_cleaned.csv`, `dev_cleaned.csv`, `test_cleaned.csv` located in `cleanedXi/`.
    *   **Sesotho:** `sesotho_cleaned.csv` . The notebook splits this into train/validation/test.
    *   **Setswana:** `setswana_cleaned.csv` . The notebook splits this into train/validation/test.
*   **Models:**
    *   Fine-tuned AfriBERTa model checkpoints are saved to Google Drive under `/content/drive/MyDrive/NLPS1/models/` within language-specific and model-strategy-specific subdirectories (e.g., `Xitsonga/afriberta_base-full/`, `Sesotho_NewData/afriberta_base-lora/`).
*   **Notebook:**
    *   `Group 17 NLP Project.ipynb`: The main Google Colab notebook containing all code and experiments.

## Notebook Workflow & How to Run

The notebook is designed to be run sequentially. Please execute cells in order.

**Part 0: Setup**
*   **(Cell A): Mount Drive and Define Paths:** Connects to Google Drive and sets file paths.
*   **(Cell B): Load Libraries for Baseline Modeling:** Imports foundational Python libraries for scikit-learn.

**Part 1: Baseline Model Training & Evaluation (Step 4)**
*   **(Cells C.1, D.1, E.1):** Xitsonga baselines (Logistic Regression, Naive Bayes, Random Forest).
*   **(Cells C.2 , D.2 , E.2 ):** Sesotho baselines.
*   **(Cells C.3 , D.3 , E.3 ):** Setswana baselines.
*   **(Cell F): Display All Baseline Results Summary:** Consolidates baseline metrics.

**Part 2: Advanced Model - AfriBERTa Fine-tuning (Step 5)**
    This project component focuses on `castorini/afriberta_base`. 

*   **Xitsonga with AfriBERTa:**
    *   **(Cell I.X2):** Prepares Xitsonga data for Hugging Face.
    *   **(Cells J.X2.1 - J.X2.7):** AfriBERTa Full Fine-tuning.
    *   **(Cells J.X2.8 - J.X2.11):** AfriBERTa LoRA Fine-tuning.
*   **Sesotho  with AfriBERTa:**
    *   **(Cell I.S1):** Prepares NEW Sesotho data for Hugging Face.
    *   **(Cells J.S2.1 - J.S2.7):** AfriBERTa Full Fine-tuning.
    *   **(Cells J.S2.8 - J.S2.11):** AfriBERTa LoRA Fine-tuning.
*   **Setswana with AfriBERTa:**
    *   **(Cell I.T1):** Prepares NEW Setswana data for Hugging Face.
    *   **(Cells J.T1.1 - J.T1.7):** AfriBERTa Full Fine-tuning.
    *   **(Cells J.T1.8 - J.T1.11):** AfriBERTa LoRA Fine-tuning.
*   **(Cell K.0 - K.0.1):** Debugging and correction cells for the `all_advanced_results` dictionary (important for ensuring summary table accuracy).
*   **(Cell K.1): Comprehensive Results Summary Table:** Displays all baseline and AfriBERTa results.

**Part 3: Data Augmentation for Sesotho (Step 6)**
Focuses on improving the best AfriBERTa configuration for Sesotho (Full FT on New Data).

*   **(Cell L.S.1):** Setup for back-translation using Helsinki-NLP models for Sesotho.
*   **(Cell L.S.2):** Applies back-translation to the NEW Sesotho training data.
*   **(Cell L.S.3):** Re-trains AfriBERTa Full FT on the augmented NEW Sesotho data and evaluates.

**Part 4: Model Evaluation & Explainability (SHAP) for Champion Sesotho Model (Step 7)**
Analyzes the AfriBERTa Full FT model trained on augmented NEW Sesotho data.

*   **(Cell M.1): Install SHAP & Load Champion Model:** Loads the specified fine-tuned checkpoint.
    *   **Note:** The `CHAMPION_MODEL_CHECKPOINT_PATH` in this cell must point to the correct saved checkpoint from the augmented training run (e.g., from Cell L.S.3).
*   **(Cell M.2): Create SHAP Explainer & Explain Predictions:** Generates SHAP plots for selected instances.
*   **(Cell M.3): Predictions & Confusion Matrix:** Provides a detailed classification report and confusion matrix using pipeline-based inference.

## Key Libraries & Tools
*   Python 3.x
*   Pandas
*   Scikit-learn
*   Hugging Face Libraries:
    *   `transformers` (for models, tokenizers, Trainer, pipelines)
    *   `datasets` (for data handling)
    *   `evaluate` (for metrics)
    *   `PEFT` (for LoRA)
*   `SHAP` (for model explainability)
*   `Matplotlib` & `Seaborn` (for plotting)

## Running the Notebook
1.  **Mount Google Drive:** When prompted by Cell A.
2.  **Verify Data Paths:** Ensure cleaned `.csv` files for all three languages are in the Google Drive locations specified in Cell A.
3.  **Run Sequentially:** Execute cells in the order presented in the notebook.
4.  **GPU Recommended:** A GPU runtime in Google Colab is **highly recommended** for the AfriBERTa fine-tuning sections (J-series, L-series) and SHAP analysis to complete efficiently.
5.  **Google Drive Storage:** Ensure sufficient Google Drive space is available for saving model checkpoints. If quota issues arise, free up space or (for temporary runs) modify output directories in `TrainingArguments` to use Colab's local `/content/` storage (note: files in `/content/` are ephemeral).

## Expected Outputs & Key Findings
*   Performance metrics (Accuracy, F1 Macro, Precision, Recall) for baseline and AfriBERTa models across Xitsonga, Sesotho, and Setswana.
*   Summary tables comparing different modeling approaches.
*   Saved checkpoints of fine-tuned AfriBERTa models in the specified Google Drive directories.
*   SHAP explanation plots illustrating feature importance for selected predictions of the champion Sesotho model.
*   A detailed classification report and confusion matrix for the champion Sesotho model, revealing its performance characteristics and biases in pipeline-based inference.
*   **Key Findings Summary:**
    *   AfriBERTa (Full Fine-tuning) generally outperformed baselines and its LoRA (r=8) counterpart across the three languages on the provided datasets.
    *   Data augmentation (back-translation) provided a notable improvement for the AfriBERTa Full FT model on Sesotho.
    *   The champion Sesotho model (AfriBERTa Full FT + Augmentation), while achieving a high F1 score in batched `Trainer` evaluation, exhibited a strong bias towards the majority 'negative' class when evaluated via single-instance pipeline inference, highlighting challenges with class imbalance.
    *   Performance varied by language, with Sesotho generally yielding the highest F1 scores with AfriBERTa.# Group-17-NLP-project
