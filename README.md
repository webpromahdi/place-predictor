# PlacePredictor

PlacePredictor is a Streamlit-based student placement prediction system. It predicts whether a student is likely to be **Placed** or **Not Placed** using a trained machine learning pipeline.

## Final Selected Model

**Naive Bayes** is the final selected model for this project.

The app also shows:

- Placement prediction: Placed / Not Placed
- Placement probability score
- AI-style explanation for why the student is predicted as Placed or Not Placed
- Skill suggestions needed to improve placement chance
- Examiner dashboard with model comparison table and EDA charts

## Dataset

The project uses the student placement dataset.

Important preprocessing decisions:

- `sl_no` is removed because it is only an ID column.
- `salary` is removed because it is available only after placement and causes data leakage.
- `status` is used as the target variable.

## Models Compared

The training script compares six independent classification models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine
5. K-Nearest Neighbors
6. Naive Bayes

All models are trained independently using the same preprocessing and evaluation approach.

## Tech Stack

Python, Streamlit, scikit-learn, imbalanced-learn, pandas, matplotlib, seaborn

## Project Structure

```text
PlacePredictor_Final_Complete_Project/
│
├── app.py
├── train_model.py
├── placement_model.pkl
├── model_results.csv
├── naive_bayes_classification_report.txt
├── requirements.txt
├── README.md
├── .gitignore
│
└── dataset/
    └── Placement_Data_Full_Class.csv
```

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train model:

```bash
python train_model.py
```

Run Streamlit app:

```bash
python -m streamlit run app.py
```

## Project Summary

PlacePredictor is a machine learning-based student placement prediction system. The system predicts whether a student is likely to be Placed or Not Placed using a trained Naive Bayes classification model.

The application takes student academic, personal, and employability-related information as input through a Streamlit web interface. After prediction, it displays the placement result, placement probability, and an AI-based explanation.

If the student is predicted as Not Placed, the system explains possible weak factors such as no work experience, low employability test score, low degree percentage, low SSC/HSC percentage, or low MBA percentage. It also suggests skills needed to improve placement chances, including internship experience, aptitude improvement, communication skills, interview preparation, CV building, and project work.

If the student is predicted as Placed, the system explains the positive factors that supported the prediction and provides further improvement suggestions for better career preparation.

The project also includes an examiner dashboard that shows model performance, model comparison results, and exploratory data analysis charts.
