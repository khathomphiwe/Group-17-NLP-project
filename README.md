# Enhancing Sentiment Analysis for Low-Resource African Languages Using Transfer Learning and Data Augmentation

**Project by: Selloane Chalale, Khathophiwe Nemutudi, and Mulisa Thovhale (Group 17)**

**Date of Submission: 19 June 2025**

## Project Overview

This project addresses the critical need for effective sentiment analysis tools in underrepresented African languages. We focus on **Xitsonga, Sesotho, and Setswana**, aiming to:

1.  Preprocess raw text data for these languages to prepare them for machine learning.
2.  Establish robust baseline performance for sentiment classification using traditional machine learning models.
3.  Explore the efficacy of transfer learning with the Africa-centric **AfriBERTa** pre-trained model.
4.  Compare different fine-tuning strategies: **Full Fine-tuning** vs. parameter-efficient **LoRA Fine-tuning**.
5.  Investigate the impact of **data augmentation** (back-translation) on model accuracy.
6.  Employ model **explainability methods (SHAP)** to understand model predictions, errors, and potential biases.

This repository contains:
*   Individual Jupyter notebooks for pre cleaning preprocessing raw data for each language (`Load_and _clean_data[Sesotho].ipynb , Load_and _clean_data[Setswana].ipynb , Load_and _clean_data[Xitsonga].ipynb`).
*   Folder with cleaned dataset(NLP1).
*   A main Google Colab notebook (`Group_17_NLP_Project.ipynb`) demonstrating the experimental pipeline from loading these pre-cleaned datasets through model training, evaluation, and explainability analysis.

##  Part 1: Data Preprocessing Pipeline

This section details the initial data preparation steps performed to clean and structure the raw text data for each language. These preprocessing steps are typically executed by separate scripts/notebooks, producing the "cleaned" datasets used in the main modeling notebook.

**Core Preprocessing Workflow (Applied per language with specific adaptations):**

1.  **Setup and Imports:** Loads necessary libraries (pandas, NLTK, scikit-learn, visualization tools, imbalanced-learn).
2.  **Data Loading and Parsing:** Reads raw data from source files (`.csv`, `.tsv`, or `.txt`).
3.  **Text Cleaning:**
    *   Convert text to lowercase.
    *   Remove URLs, website links, and user mentions (e.g., @username).
    *   Remove all non-alphabetic characters (keeping only letters and spaces). 
    *   Normalize whitespace.
4.  **Tokenization:** Cleaned text is tokenized into individual words using NLTK's `punkt` tokenizer.
5.  **Data Visualization (During Preprocessing Stage):**
    *   Sentiment label distribution (Count Plot).
    *   Word Cloud of frequent words.
    *   Bar Plot of top common words.
6.  **Feature Engineering (for initial exploration/balancing):** Text converted to numerical features using TfidfVectorizer (this is distinct from TF-IDF used for baselines in the main modeling notebook).
7.  **Handling Class Imbalance (During Preprocessing Stage):** SMOTE (Synthetic Minority Over-sampling Technique) was used to oversample minority classes in the raw data processing stage, aiming to create more balanced datasets for subsequent modeling.
8.  **Train-Test Split (During Preprocessing Stage for single files):** For datasets provided as single files (Setswana, Sesotho), they were split into training (80%) and testing (20%) sets after SMOTE. Xitsonga used its provided train/dev/test splits.
9.  **Save Cleaned Data:** The processed DataFrames (now including features like `cleaned_text`, `label_numeric`, etc.) are exported to `.csv` files.

**Dataset-Specific Preprocessing Details:**

*   **Setswana (`setswana_processor.py`):**
    *   **Input:** `data/setswana_tweets.csv` (multi-language, filtered for 'Setswana' in `predict_name` column). Uses `sentence` for text, `Final_Label` for sentiment.
    *   **Output:** `output/cleaned_setswana_tweets.csv` (This is the `setswana_cleaned.csv` used in the main modeling notebook).
*   **Xitsonga (`xitsonga_processor.py`):**
    *   **Input:** `data/train.tsv`, `data/dev.tsv`, `data/test.tsv`.
    *   **Description:** Loads and cleans each TSV file. Uses `tweet` for text, `label` for sentiment.
    *   **Output:** `output/train_cleaned.csv`, `output/dev_cleaned.csv`, `output/test_cleaned.csv` (These are used by the main modeling notebook, found in `cleanedXi/`).
*   **Sesotho (`sesotho_processor.py`):**
    *   **Input:** `data/sesotho_uncleaned_data.txt` (custom format: sentence on one line, numeric label (-1,0,1) on the next).
    *   **Description:** Parses the custom format. Includes SMOTE with robustness for small minority classes.
    *   **Output:** `output/cleaned_sesotho_data.csv` (This is the `sesotho_cleaned.csv` used in the main modeling notebook).

**Expected Preprocessing Output Structure (for main modeling notebook):**
The `output/` directory (or `NLPS1/Cleaned/` in Google Drive) should contain:
*   `cleanedXi/train_cleaned.csv`
*   `cleanedXi/dev_cleaned.csv`
*   `cleanedXi/test_cleaned.csv`
*   `sesotho_cleaned.csv` (with columns like `cleaned_text`, `label_numeric`)
*   `setswana_cleaned.csv` (with columns like `cleaned_sentences`, `Final_Label`)

## Part 2: Sentiment Analysis Modeling & Evaluation (`Group_17_NLP_Project.ipynb`)

This Google Colab notebook takes the cleaned datasets generated in Part 1 and performs sentiment analysis experiments.

**Notebook Workflow & How to Run:**
The notebook is designed to be run sequentially.

**Section 0: Setup**
*   **(Cell A): Mount Drive and Define Paths:** Connects to Google Drive and sets paths to the **cleaned** data files (e.g., `/content/drive/MyDrive/NLPS1/Cleaned/`). The Folder with the cleaned data is uploaded with the name NLP1
*   **(Cell B): Load Libraries for Baseline Modeling:** Imports foundational Python libraries.

**Section 1: Baseline Model Training & Evaluation (Step 4)**
*   **(Cells C.1, D.1, E.1):** Xitsonga baselines (Logistic Regression, Naive Bayes, Random Forest).
*   **(Cells C.2, D.2, E.2):** Sesotho (New Data from preprocessing) baselines.
*   **(Cells C.3, D.3, E.3):** Setswana (New Data from preprocessing) baselines.
*   **(Cell F): Display All Baseline Results Summary:** Consolidates baseline metrics.

**Section 2: Advanced Model - AfriBERTa Fine-tuning (Step 5)**
    This project component focuses on `castorini/afriberta_base`.

*   **Xitsonga with AfriBERTa:**
    *   **(Cell I.X2):** Prepares Xitsonga data for Hugging Face.
    *   **(Cells J.X2.1 - J.X2.7):** AfriBERTa Full Fine-tuning.
    *   **(Cells J.X2.8 - J.X2.11):** AfriBERTa LoRA Fine-tuning.
*   **Sesotho (New Data) with AfriBERTa:**
    *   **(Cell I.S1):** Prepares Sesotho data for Hugging Face.
    *   **(Cells J.S2.1 - J.S2.7):** AfriBERTa Full Fine-tuning.
    *   **(Cells J.S2.8 - J.S2.11):** AfriBERTa LoRA Fine-tuning.
*   **Setswana (New Data) with AfriBERTa:**
    *   **(Cell I.T1):** Prepares Setswana data for Hugging Face.
    *   **(Cells J.T1.1 - J.T1.7):** AfriBERTa Full Fine-tuning.
    *   **(Cells J.T1.8 - J.T1.11):** AfriBERTa LoRA Fine-tuning.
*   **(Cell K.0 - K.0.1):** Debugging and correction cells for `all_advanced_results`.
*   **(Cell K.1): Comprehensive Results Summary Table:** Displays all baseline and AfriBERTa results.

**Section 3: Data Augmentation for Sesotho (Step 6)**
Focuses on improving the best AfriBERTa configuration for Sesotho.

*   **(Cell L.S.1):** Setup for back-translation using Helsinki-NLP models for Sesotho training data.
*   **(Cell L.S.2):** Applies back-translation to Sesotho training data.
*   **(Cell L.S.3):** Re-trains AfriBERTa Full FT on the augmented Sesotho data and evaluates.

**Section 4: Model Evaluation & Explainability (SHAP) for Champion Sesotho Model (Step 7)**
Analyzes the AfriBERTa Full FT model trained on augmented Sesotho data.

*   **(Cell M.1): Install SHAP & Load Champion Model.**
    *   **Note:** The `CHAMPION_MODEL_CHECKPOINT_PATH` must point to the correct saved checkpoint from the augmented training run (e.g., from Cell L.S.3).
*   **(Cell M.2): Create SHAP Explainer & Explain Predictions.**
*   **(Cell M.3): Predictions & Confusion Matrix.**

## Key Libraries & Tools (Combined for Preprocessing & Modeling)
*   Python 3.x
*   Pandas
*   NLTK (for tokenization)
*   Matplotlib & Seaborn (for plotting)
*   Wordcloud
*   Scikit-learn (for TF-IDF, baselines, metrics, train-test split)
*   Imbalanced-learn (for SMOTE in preprocessing scripts)
*   Hugging Face Libraries:
    *   `transformers` (for models, tokenizers, Trainer, pipelines)
    *   `datasets` (for data handling)
    *   `evaluate` (for metrics)
    *   `PEFT` (for LoRA)
*   `SHAP` (for model explainability)

## Running the Code

**For Preprocessing (if running separate scripts):**
1.  **Prerequisites:** Python, pip.
2.  **Install Dependencies:** `pip install pandas nltk matplotlib seaborn wordcloud scikit-learn imbalanced-learn`
3.  **NLTK Data:** Download `punkt` tokenizer (usually automatic on first NLTK run).
4.  **Data Files:** Place raw input data into a `data/` directory as specified in the "Dataset-Specific Preprocessing Details" section.
5.  **Run Scripts:** Execute `python <script_name>.py`. Outputs will be in `output/`.

**For Main Modeling Notebook (`Group_17_NLP_Project.ipynb`):**
1.  **Upload Cleaned Data:** Ensure the **cleaned** `.csv` files (generated by the preprocessing scripts, e.g., `cleaned_sesotho_data.csv`, `cleaned_setswana_tweets.csv`, and the Xitsonga files) are uploaded to the Google Drive path specified in Cell A (e.g., `/content/drive/MyDrive/NLPS1/Cleaned/`).
2.  **Mount Google Drive:** When prompted by Cell A.
3.  **Run Sequentially:** Execute cells in the order presented.
4.  **GPU Recommended:** A GPU runtime in Google Colab is **highly recommended** for AfriBERTa fine-tuning and SHAP analysis.
5.  **Google Drive Storage:** Ensure sufficient Google Drive space for model checkpoints.

## Expected Outputs & Key Findings (from Modeling Notebook)
*   Performance metrics (Accuracy, F1 Macro, etc.) for baselines and AfriBERTa across all three languages.
*   Summary tables comparing model performances.
*   Saved AfriBERTa model checkpoints in Google Drive.
*   SHAP explanation plots for the champion Sesotho model.
*   Classification report and confusion matrix for the champion Sesotho model, highlighting inference behavior and biases.
*   **Key Findings Summary:**
    *   AfriBERTa (Full Fine-tuning) generally outperformed baselines and its LoRA (r=8) configuration.
    *   Data augmentation (back-translation) provided a notable improvement for AfriBERTa Full FT on Sesotho.
    *   The champion Sesotho model, while achieving a high F1 score in `Trainer` evaluation (~0.666), exhibited a strong bias towards the majority 'negative' class in pipeline-based inference (F1 Macro ~0.28), emphasizing the challenge of class imbalance.
    *   Performance varied by language, with Sesotho generally yielding the highest F1 scores with AfriBERTa.
