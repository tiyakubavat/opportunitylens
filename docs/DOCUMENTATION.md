# OpportunityLens - Technical Documentation

Simulated scholarship-shortlisting scenario. Research prototype, not for real decisions.

## Repository structure
```
app.py              Streamlit app: data, model, explanations, checks, audit, receipt
requirements.txt    Python dependencies
README.md           Overview, install, usage, dataset, model, model screenschots, roadmap, credits
docs/               This file
```

## How the app works
1. **Data (`make_data`)**: 3,000 synthetic applicants generated with a fixed seed. Labels come from a documented noisy weighted score of merit (GPA, test, essay, recommendation, extracurriculars) and financial need (family income). Rural applicants have lower simulated test scores and extracurricular access. `region` is never given to the model.
2. **Model (`train`)**: stratified 75/25 split, standardized features, logistic regression and gradient boosting compared by 5-fold cross-validated ROC-AUC. The simpler model is selected unless boosting is better by more than 0.01 AUC. Decision threshold is 0.5.
3. **Explanations**: SHAP values in log-odds. Local view shows one applicant's contributions. Global view shows mean absolute SHAP over 300 test applicants.
4. **Data checks (`data_checks`)**:
   - High: test score or income missing/0; essay or recommendation 0
   - Medium: GPA >= 3.5 with a test score below 40
   - Low: value outside the 1st-99th percentile of the training data
5. **Comparison**: what-if profile B versus the current profile A, plus the three closest synthetic applicants (Euclidean distance on standardized features) with a different model outcome.
6. **Group audit**: selection rate, true-positive rate and precision by region. Selection-rate ratio below 0.80 is shown as a warning signal.
7. **Validation**: baseline comparison, ROC-AUC, Brier score, calibration curve, injected-problem detection test (zero-coded missing value, silent median imputation, simulated label bias), and a SHAP-versus-known-rules check.
8. **Decision Receipt**: five sections (What happened? Why? Was it consistent? What can be challenged? What are the limits?). Export as HTML, Markdown or PDF. An optional applicant note requests human review.

## Interpreting the outputs
- SHAP contributions describe model behaviour, not causation.
- A changed prediction after editing an input is a signal, not proof of discrimination.
- Group gaps need context. Here, part of the gap is built into the synthetic data.
- Metrics on synthetic data do not show validity for real decisions.
- Known miss: silently imputed missing values are not detectable by the current rules.

## Reproducibility
Seeds are fixed (`make_data` seed 42, split `random_state=1`). Results can still differ slightly across library versions.

## Running locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
