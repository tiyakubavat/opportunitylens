# OpportunityLens

**Understand the decision. Examine the difference. Question the outcome.**

An explainable-AI decision-auditing prototype, demonstrated on a **simulated** scholarship-shortlisting scenario.
Research and demonstration prototype only. It is **not** a real scholarship system and **not** proof that any institution discriminated against anyone.

- **Live demo:** `(https://opportunitylens-qyquhwnql5icnqbgzkdb94.streamlit.app)`

## Team
- **Team name:** `POWERPUFF GURLS`
- **Members:** `TIYA.S.KUBAVAT`, `CHARU NETHRA`, `SHRUTIGHA` 

## Selected problem statement
**Challenge 06 - Explainable AI (XAI): Fairness & Accountability in Social Systems.**

## Problem description
AI-assisted systems increasingly influence scholarships, admissions, recruitment and financial support. The people affected often see only an outcome score. They cannot tell how it was produced, what information may be wrong, or what to ask a human reviewer. A prediction score alone is not an explanation, and an explanation alone is not recourse.

## Proposed solution
OpportunityLens closes the gap between *explaining* a prediction and *scrutinizing* it. For one applicant it shows why the model predicted what it did, flags input data that may need verification, compares what-if profiles and similar cases, audits model performance and group-level gaps, and produces a **Decision Receipt**: a human-readable record with questions for human review and a clear statement of limitations.

We do not claim that explainability or fairness toolkits are new. Our contribution is the focused, applicant-centred workflow that combines them into one reviewable record.

## Key features
- **Local and global SHAP explanations** with plain-language wording
- **Input-data checks:** missing or zero values, inconsistent combinations, out-of-range values
- **Controlled what-if comparison** (profile A vs B) and closest synthetic applicants with a different outcome
- **Group-level audit** (Urban vs Rural): selection rate, true-positive rate, precision, selection-rate ratio against a 0.80 warning rule
- **Validation tab:** model comparison by cross-validation, ROC-AUC, Brier score, calibration, majority-class baseline, injected-problem detection test (including misses), and a SHAP-vs-known-rules check
- **Decision Receipt** with five sections: What happened? Why? Was it consistent? What can be challenged? What are the limits? Downloadable as HTML, Markdown or PDF, with an optional applicant request for human review

## Technology stack
Python, pandas, NumPy, scikit-learn, SHAP, Streamlit, Plotly, fpdf2, Git/GitHub.

## Dataset information
All data is **synthetic** (3,000 simulated applicants, generated in `make_data()` with a fixed seed).
- Features: GPA, test score, essay score, recommendation, extracurriculars, family income.
- Labels: noisy weighted merit-plus-financial-need score above a cutoff. The weights are documented in the code.
- Built-in structural gap: simulated rural applicants have lower test scores and extracurricular access.
- `region` is **not** a model input. It is used only for the group audit.
- No real scholarship records or personal data are used.

## Model information
- Candidates: logistic regression and gradient boosting, compared by 5-fold cross-validated ROC-AUC. The simpler model is preferred unless the complex one is clearly better.
- Selected in our run: **logistic regression** (stratified 75/25 train/test split, standardized features).
- Explanations: `shap.LinearExplainer` (or `TreeExplainer` if boosting wins). If SHAP is unavailable, the app falls back to equivalent linear contributions.
- Decision threshold: 0.5.
- In our run: test accuracy 88.0%, ROC-AUC 0.947. These are synthetic-data results: the labels come from a rule the model can learn, so they do **not** show validity for real decisions.

## Methodology
1. Generate synthetic applicants and document how labels were created.
2. Train and evaluate models on a held-out test set.
3. Explain predictions locally and globally with SHAP, and check that SHAP's top drivers match the known generating rules.
4. Run controlled checks: injected missing values, injected simulated bias, what-if edits.
5. Combine the real prediction, explanation, checks, evaluation context and limitations into the Decision Receipt.

Safeguards: feature importance is not causation; a prediction change after editing an input is not proof of discrimination; group gaps are signals for investigation, not findings.

## Architecture / workflow
```
Applicant profile (sidebar)
        |
Data checks --> flags (missing / inconsistent / out-of-range)
        |
Model (scaler + classifier) --> prediction + score
        |
SHAP explanation --> local + global
        |
Comparison + group audit + validation
        |
Decision Receipt (HTML / MD / PDF) --> human review
```
Single Streamlit app (`app.py`): data generation and training are cached, so the interface responds immediately.

## Installation instructions
Requires Python 3.11 or 3.12.
```bash
git clone https://github.com/YOUR_USERNAME/opportunitylens.git
cd opportunitylens
pip install -r requirements.txt
```

## Usage instructions
```bash
streamlit run app.py
```
1. Pick a scenario in the sidebar (default: "Demo: Asha") or enter your own profile.
2. Read the prediction, then open the tabs in order: **Explain**, **Check data**, **Compare**, **Model & groups**, **Decision Receipt**, **Validation**.
3. In **Decision Receipt**, optionally type what a reviewer should check, then download the receipt.

## Screenshots


![Home]<img width="1867" height="970" alt="image" src="https://github.com/user-attachments/assets/c842e1f2-25d5-4c22-b64c-a710d3ca2363" />

![Explain]<img width="1315" height="687" alt="image" src="https://github.com/user-attachments/assets/49904c0e-dd6b-4fb6-9e33-a188205e8a95" />
<img width="1387" height="577" alt="image" src="https://github.com/user-attachments/assets/b44c4183-8acc-4210-b3f1-36f72168ee4a" />


![Decision Receipt]<img width="727" height="810" alt="image" src="https://github.com/user-attachments/assets/42555fdb-c0ec-4de2-8550-d2894d038e46" />


## Limitations
Synthetic data only. One scenario, one simple model, and no real-stakeholder evaluation. Silent imputation of missing values cannot be detected by the current rules. Not for actual eligibility or scholarship decisions.

## Future research roadmap
- Multiple decision domains (admissions, hiring, loans)
- More model families and explanation techniques
- Broader fairness auditing with additional protected attributes and intersections
- Secure reviewer workflows and audit history
- User studies with real stakeholders, using appropriately governed data

## Credits / attributions
- [SHAP](https://github.com/shap/shap), [scikit-learn](https://scikit-learn.org), [Streamlit](https://streamlit.io), [Plotly](https://plotly.com), [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org), [fpdf2](https://github.com/py-pdf/fpdf2)
- AI assistance: Claude (Anthropic) was used to help scaffold code and documentation. 

