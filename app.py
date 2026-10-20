"""OpportunityLens - explainable-AI decision-auditing prototype (SYNTHETIC scholarship scenario).
Run:  streamlit run app.py
NOT a real scholarship system. All data is simulated."""
import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import shap
    HAS_SHAP = True
except Exception:  # fallback keeps the demo alive if shap fails to install
    HAS_SHAP = False

try:
    from fpdf import FPDF
    HAS_PDF = True
except Exception:
    HAS_PDF = False

st.set_page_config(page_title="OpportunityLens", page_icon="🔍", layout="wide")

FEATURES = ["gpa", "test_score", "essay_score", "recommendation", "extracurriculars", "family_income_k"]
NAMES = {"gpa": "GPA", "test_score": "Test score", "essay_score": "Essay score",
         "recommendation": "Recommendation", "extracurriculars": "Extracurriculars",
         "family_income_k": "Family income (k/yr)"}
THRESHOLD = 0.5


# ---------------------------------------------------------------- data + model
@st.cache_data
def make_data(n=3000, seed=42):
    """Synthetic applicants. Label = noisy weighted merit + financial-need score > cutoff.
    Documented structural gap: rural applicants have lower test scores / extracurricular access.
    'region' is NOT a model input; it is kept only for the group-level audit."""
    r = np.random.default_rng(seed)
    region = r.choice(["Urban", "Rural"], n, p=[0.6, 0.4])
    rural = (region == "Rural").astype(float)
    d = pd.DataFrame({
        "gpa": np.clip(r.normal(3.0, 0.5, n), 1.5, 4.0),
        "test_score": np.clip(r.normal(65, 15, n) - 4 * rural, 5, 100),
        "essay_score": np.clip(r.normal(6, 1.8, n), 1, 10),
        "recommendation": np.clip(r.normal(6.5, 1.7, n), 1, 10),
        "extracurriculars": np.clip(r.normal(5, 2, n) - 1.5 * rural, 0, 10),
        "family_income_k": np.clip(r.lognormal(3.8, 0.6, n), 5, 200),
        "region": region})
    need = 1 - np.clip(d.family_income_k, 5, 150) / 150
    score = (0.9 * (d.gpa - 3) / 0.5 + 0.7 * (d.test_score - 65) / 15 + 0.5 * (d.essay_score - 6) / 1.8
             + 0.4 * (d.recommendation - 6.5) / 1.7 + 0.3 * (d.extracurriculars - 5) / 2
             + 0.6 * (need - 0.7) / 0.2 + r.normal(0, 0.6, n))
    d["y"] = (score > 0.6).astype(int)
    return d


@st.cache_resource
def train():
    d = make_data()
    tr, te = train_test_split(d, test_size=0.25, random_state=1, stratify=d.y)
    sc = StandardScaler().fit(tr[FEATURES])
    Xtr, Xte = sc.transform(tr[FEATURES]), sc.transform(te[FEATURES])
    cands = {"Logistic regression": LogisticRegression(max_iter=1000)}
    if HAS_SHAP:
        cands["Gradient boosting"] = GradientBoostingClassifier(random_state=0)
    cv = {n: float(cross_val_score(m, Xtr, tr.y, cv=5, scoring="roc_auc").mean()) for n, m in cands.items()}
    fitted = {n: m.fit(Xtr, tr.y) for n, m in cands.items()}
    test_auc = {n: float(roc_auc_score(te.y, m.predict_proba(Xte)[:, 1])) for n, m in fitted.items()}
    best = "Logistic regression"  # prefer the simpler model unless the complex one is clearly better
    if "Gradient boosting" in cv and cv["Gradient boosting"] > cv["Logistic regression"] + 0.01:
        best = "Gradient boosting"
    model = fitted[best]
    te = te.copy()
    te["prob"] = model.predict_proba(Xte)[:, 1]
    te["pred"] = (te.prob >= THRESHOLD).astype(int)
    if HAS_SHAP:
        expl = shap.LinearExplainer(model, Xtr) if best == "Logistic regression" else shap.TreeExplainer(model)
    else:
        expl = None
    base = float(model.decision_function(Xtr).mean())
    if best != "Logistic regression" and expl is not None:
        base = float(np.ravel(expl.expected_value)[0])
    allp = d.copy()
    allp["prob"] = model.predict_proba(sc.transform(d[FEATURES]))[:, 1]
    allp["pred"] = (allp.prob >= THRESHOLD).astype(int)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, tr.y)
    fp, mp = calibration_curve(te.y, te.prob, n_bins=8, strategy="quantile")
    metrics = dict(accuracy=accuracy_score(te.y, te.pred), precision=precision_score(te.y, te.pred),
                   recall=recall_score(te.y, te.pred), f1=f1_score(te.y, te.pred),
                   cm=confusion_matrix(te.y, te.pred), auc=float(roc_auc_score(te.y, te.prob)),
                   brier=float(brier_score_loss(te.y, te.prob)),
                   base_acc=accuracy_score(te.y, dummy.predict(Xte)), cal=(mp, fp))
    lo, hi = tr[FEATURES].quantile(0.01), tr[FEATURES].quantile(0.99)
    return dict(sc=sc, model=model, best=best, cv=cv, test_auc=test_auc, te=te, tr=tr, expl=expl, all=allp,
                m=metrics, base=base, lo=lo, hi=hi, ntrain=len(tr))

M = train()


def predict(row):
    xs = M["sc"].transform(pd.DataFrame([row])[FEATURES])
    return float(M["model"].predict_proba(xs)[0, 1]), xs


def contribs(xs):
    """SHAP values (log-odds) for one scaled row; falls back to coef*x (identical for a linear model)."""
    if M["expl"] is not None:
        try:
            return np.array(M["expl"].shap_values(xs)).reshape(-1)[-len(FEATURES):]
        except Exception:
            pass
    return M["model"].coef_[0] * xs[0]


def decision(p):
    return "Shortlisted" if p >= THRESHOLD else "Not shortlisted"


def data_checks(row, test_missing, income_missing):
    out = []
    if test_missing or row["test_score"] <= 0:
        out.append(("High", "Test score is missing/0 in the record. The model treats 0 as a real, very low score."))
    if income_missing or row["family_income_k"] <= 0:
        out.append(("High", "Family income is missing/0 in the record. The model treats 0 as a real income."))
    for f in ["essay_score", "recommendation"]:
        if row[f] <= 0:
            out.append(("High", f"{NAMES[f]} is 0 - likely missing."))
    if row["gpa"] >= 3.5 and 0 < row["test_score"] < 40:
        out.append(("Medium", "Very high GPA but very low test score - possible entry error or genuine mismatch."))
    for f in FEATURES:
        v = row[f]
        if v > 0 and (v < M["lo"][f] or v > M["hi"][f]):
            out.append(("Low", f"{NAMES[f]} = {v:.1f} lies outside the typical training range "
                               f"({M['lo'][f]:.1f}-{M['hi'][f]:.1f}); the model is extrapolating."))
    return out


def plain(f, v, c):
    mag = "strongly" if abs(c) > 1.5 else "moderately" if abs(c) > 0.5 else "slightly"
    return (f"{NAMES[f]} ({v:.1f}) {mag} {'raised' if c > 0 else 'lowered'} the chance of being shortlisted "
            f"({c:+.2f} log-odds)")


# ---------------------------------------------------------------- sidebar
ASHA = "Demo: Asha (test score missing in record)"
PRESETS = {
    ASHA: dict(gpa=3.6, test=78.0, essay=7.0, rec=8.0, extra=4.0, income=40.0,
               test_missing=True, income_missing=False, region="Rural"),
    "Strong, complete profile": dict(gpa=3.8, test=85.0, essay=8.0, rec=8.0, extra=7.0, income=30.0,
                                     test_missing=False, income_missing=False, region="Urban"),
    "Borderline profile": dict(gpa=3.0, test=65.0, essay=6.0, rec=6.5, extra=5.0, income=45.0,
                               test_missing=False, income_missing=False, region="Urban"),
    "Custom": None}


def apply_preset():
    p = PRESETS.get(st.session_state.preset)
    if p:
        for k, v in p.items():
            st.session_state[k] = v


for k, v in PRESETS[ASHA].items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.title("🔍 OpportunityLens")
    st.caption("Synthetic scholarship-shortlisting demo. Not for real decisions.")
    st.selectbox("Load a scenario", list(PRESETS), key="preset", on_change=apply_preset)
    st.slider("GPA (1.5-4.0)", 1.5, 4.0, step=0.05, key="gpa")
    st.slider("Test score (0-100)", 0.0, 100.0, step=1.0, key="test")
    st.checkbox("Test score MISSING in record (pipeline stores 0)", key="test_missing")
    st.slider("Essay score (0-10)", 0.0, 10.0, step=0.5, key="essay")
    st.slider("Recommendation (0-10)", 0.0, 10.0, step=0.5, key="rec")
    st.slider("Extracurriculars (0-10)", 0.0, 10.0, step=0.5, key="extra")
    st.slider("Family income (k per year)", 0.0, 200.0, step=1.0, key="income")
    st.checkbox("Income MISSING in record (pipeline stores 0)", key="income_missing")
    st.radio("Region (audit only - NOT a model input)", ["Urban", "Rural"], key="region", horizontal=True)

S = st.session_state
row = dict(gpa=S.gpa, test_score=0.0 if S.test_missing else S.test, essay_score=S.essay, recommendation=S.rec,
           extracurriculars=S.extra, family_income_k=0.0 if S.income_missing else S.income)
prob, xs = predict(row)
sv = contribs(xs)
dec = decision(prob)
checks = data_checks(row, S.test_missing, S.income_missing)

# what-if: same applicant with the missing values resolved (entered values used)
resolved = dict(row, test_score=S.test, family_income_k=S.income)
p_res, _ = predict(resolved)
changed_by_fix = bool(S.test_missing or S.income_missing) and decision(p_res) != dec

# nearest synthetic applicants with a different model outcome
allx = M["sc"].transform(M["all"][FEATURES])
dist = np.linalg.norm(allx - xs, axis=1)
M_all = M["all"].assign(distance=dist)
neighbor = M_all[M_all.pred != int(prob >= THRESHOLD)].sort_values("distance").head(3)

# group audit
te = M["te"]
rows = []
for g, s in te.groupby("region"):
    rows.append({"Group": g, "Applicants": len(s), "Label rate": s.y.mean(), "Selection rate": s.pred.mean(),
                 "True-positive rate": s[s.y == 1].pred.mean(), "Precision": s[s.pred == 1].y.mean()})
grp = pd.DataFrame(rows)
di = grp["Selection rate"].min() / grp["Selection rate"].max()
low_g = grp.loc[grp["Selection rate"].idxmin(), "Group"]

# ---------------------------------------------------------------- header
st.title("OpportunityLens")
st.caption("Understand the decision. Examine the difference. Question the outcome.  |  "
           "Simulated data - research prototype, not proof of discrimination.")
c1, c2, c3 = st.columns(3)
c1.metric("Model prediction", dec)
c2.metric("Prediction score", f"{prob:.1%}", f"threshold {THRESHOLD:.0%}", delta_color="off")
c3.metric("Data issues flagged", len(checks))
if changed_by_fix:
    st.warning(f"With the missing value(s) resolved, the model output changes to **{decision(p_res)}** "
               f"({p_res:.1%}). This is a warning signal worth human review, not proof of unfairness.")

t1, t2, t3, t4, t5, t6 = st.tabs(["1 Explain", "2 Check data", "3 Compare", "4 Model & groups", "5 Decision Receipt", "6 Validation"])

# ---------------------------------------------------------------- explain
with t1:
    st.subheader("Why did the model predict this? (local SHAP)")
    order = np.argsort(np.abs(sv))
    fig = go.Figure(go.Bar(x=sv[order], y=[NAMES[FEATURES[i]] for i in order], orientation="h",
                           marker_color=["#2e7d32" if v > 0 else "#c62828" for v in sv[order]]))
    fig.update_layout(height=320, xaxis_title="Contribution to log-odds of shortlisting",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Model: {M['best']}. Baseline log-odds {M['base']:+.2f} + contributions {sv.sum():+.2f} = "
               f"{M['base'] + sv.sum():+.2f} (model output). Explainer: "
               f"{'SHAP LinearExplainer' if M['expl'] is not None else 'linear contributions (SHAP unavailable)'}.")
    for i in np.argsort(-np.abs(sv))[:3]:
        st.write("• " + plain(FEATURES[i], row[FEATURES[i]], sv[i]))
    st.subheader("Global: which features drive the model overall?")
    sample = M["sc"].transform(te[FEATURES].head(300))
    raw = np.array(M["expl"].shap_values(sample)) if M["expl"] is not None else M["model"].coef_[0] * sample
    gdf = pd.DataFrame({"Feature": [NAMES[f] for f in FEATURES], "Mean |SHAP|": np.abs(raw).mean(0)}
                       ).sort_values("Mean |SHAP|")
    st.plotly_chart(px.bar(gdf, x="Mean |SHAP|", y="Feature", orientation="h", height=300), width="stretch")
    st.info("Feature importance describes the model's behaviour. It does not prove that a feature caused the outcome.")

# ---------------------------------------------------------------- data checks
with t2:
    st.subheader("Information that may need verification")
    if not checks:
        st.success("No input problems detected by the automated rules. This does not guarantee the data is correct.")
    for sev, msg in checks:
        (st.error if sev == "High" else st.warning if sev == "Medium" else st.info)(f"**{sev}:** {msg}")
    st.write("**Record as the model saw it vs. values entered:**")
    st.dataframe(pd.DataFrame({"Feature": [NAMES[f] for f in FEATURES], "Model input": [row[f] for f in FEATURES],
                               "Value entered": [resolved[f] for f in FEATURES]}), hide_index=True, width="stretch")

# ---------------------------------------------------------------- compare
with t3:
    st.subheader("Controlled comparison: current profile (A) vs what-if profile (B)")
    st.caption("Edit B to test one change at a time. A changed output is a signal to investigate, not proof of discrimination.")
    cols = st.columns(3)
    b = {}
    for i, f in enumerate(FEATURES):
        mx = {"gpa": 4.0, "test_score": 100.0, "family_income_k": 200.0}.get(f, 10.0)
        mn = 1.5 if f == "gpa" else 0.0
        b[f] = cols[i % 3].slider(NAMES[f], mn, mx, float(np.clip(resolved[f], mn, mx)),
                                  key=f"b_{f}_{resolved[f]}")
    pb, xb = predict(b)
    svb = contribs(xb)
    ca, cb = st.columns(2)
    ca.metric("Profile A (as in record)", decision(prob), f"{prob:.1%}", delta_color="off")
    cb.metric("Profile B (what-if)", decision(pb), f"{pb - prob:+.1%} vs A")
    cmp = pd.DataFrame({"Feature": [NAMES[f] for f in FEATURES], "A": sv, "B": svb})
    st.plotly_chart(px.bar(cmp.melt("Feature", var_name="Profile", value_name="SHAP"), x="SHAP", y="Feature",
                           color="Profile", barmode="group", orientation="h", height=340), width="stretch")
    st.subheader("Closest synthetic applicants with a DIFFERENT model outcome")
    nb = neighbor[FEATURES + ["region", "prob", "pred", "distance"]].copy()
    nb["pred"] = nb.pred.map({1: "Shortlisted", 0: "Not shortlisted"})
    nb.columns = [NAMES.get(c, c) for c in nb.columns]
    st.dataframe(nb.round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------- model + groups
with t4:
    m = M["m"]
    st.subheader("Model performance (held-out synthetic test set)")
    k = st.columns(4)
    k[0].metric("Accuracy", f"{m['accuracy']:.2%}")
    k[1].metric("Precision", f"{m['precision']:.2%}")
    k[2].metric("Recall", f"{m['recall']:.2%}")
    k[3].metric("F1", f"{m['f1']:.2%}")
    st.plotly_chart(px.imshow(m["cm"], text_auto=True, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                              color_continuous_scale="Blues", height=300), width="stretch")
    st.subheader("Group-level audit: Urban vs Rural")
    st.dataframe(grp.style.format({c: "{:.1%}" for c in grp.columns[2:]}), hide_index=True, width="stretch")
    (st.warning if di < 0.8 else st.success)(
        f"Selection-rate ratio (lowest/highest) = **{di:.2f}** (group with lower rate: {low_g}). "
        "Ratios below ~0.80 are a common warning signal for further investigation.")
    st.write("**Possible proxy gap** (region is not a model input, but other features differ by region):")
    st.dataframe(make_data().groupby("region")[FEATURES].mean().rename(columns=NAMES).round(2), width="stretch")
    st.info("Group differences need context and careful interpretation. Good performance on synthetic data does "
            "not show the model is valid for real scholarship decisions.")


# ---------------------------------------------------------------- receipt
def build_receipt():
    top = [plain(FEATURES[i], row[FEATURES[i]], sv[i]) for i in np.argsort(-np.abs(sv))[:3]]
    L = ["# Decision Receipt - OpportunityLens", f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
         "SIMULATED data and model. Not a real scholarship decision.", "",
         "## 1. What happened?",
         f"- Simulated outcome: {dec}",
         f"- Prediction score: {prob:.1%} (decision threshold {THRESHOLD:.0%})",
         f"- Model: {M['best']} trained on {M['ntrain']} synthetic applicants (test ROC-AUC {M['m']['auc']:.2f})", "",
         "## 2. Why did it happen?"] + [f"- {t}" for t in top] + [
         "- These are SHAP model contributions, not proof of causation.", "",
         "## 3. Was it consistent?",
         "- Closest synthetic applicants with a different outcome:"]
    for _, r_ in neighbor.iterrows():
        L.append(f"- GPA {r_.gpa:.2f}, test {r_.test_score:.0f}, essay {r_.essay_score:.1f}, "
                 f"income {r_.family_income_k:.0f}k -> {decision(r_.prob)} ({r_.prob:.1%})")
    L += [f"- Selection-rate ratio across regions: {di:.2f} (lower group: {low_g})"
          + (" - below 0.80, investigate" if di < 0.8 else " - above 0.80"),
          f"- Test accuracy {M['m']['accuracy']:.1%}, precision {M['m']['precision']:.1%}, "
          f"recall {M['m']['recall']:.1%}, F1 {M['m']['f1']:.1%}", "",
          "## 4. What can be challenged?"]
    L += [f"- [{s_}] {m_}" for s_, m_ in checks] or ["- No automated input issues detected (does not guarantee correctness)."]
    if S.test_missing or S.income_missing:
        L.append(f"- With missing values resolved the output would be: {decision(p_res)} ({p_res:.1%})")
    L.append("- Route to review: the applicant may request a human review of the flagged items and this explanation.")
    note = (S.get("review_note", "") or "").strip()
    if note:
        L.append(f"- Applicant review request: {note}")
    L += ["", "## 5. What are the limits?",
          "- Feature importance does not prove causation.",
          "- A changed prediction after editing an input does not by itself prove discrimination.",
          "- Group differences are signals for investigation, not verified findings.",
          "- Performance on synthetic data does not show validity for real decisions.",
          "- This prototype must not be used for actual eligibility or scholarship decisions.",
          "- Reviewer should inspect: flagged inputs, the source documents, similar cases and the group gap above."]
    return "\n".join(L)


def to_html(md):
    mx = max(np.abs(sv).max(), 1e-9)
    bars = "<h3>Visual breakdown (SHAP contribution)</h3>" + "".join(
        f"<div style='display:flex;align-items:center;margin:3px 0'><span style='width:190px'>{NAMES[FEATURES[i]]}</span>"
        f"<div style='height:14px;width:{abs(sv[i]) / mx * 220:.0f}px;background:{'#2e7d32' if sv[i] > 0 else '#c62828'}'></div>"
        f"<span style='margin-left:8px'>{sv[i]:+.2f}</span></div>" for i in np.argsort(-np.abs(sv)))
    out = []
    for l in md.split("\n"):
        e = html.escape(l.strip())
        if l.startswith("## 3."):
            out.append(bars)
        if l.startswith("# "):
            out.append(f"<h1>{html.escape(l[2:])}</h1>")
        elif l.startswith("## "):
            out.append(f"<h2>{html.escape(l[3:])}</h2>")
        elif e.startswith("- "):
            out.append(f"<li>{e[2:]}</li>")
        elif e:
            out.append(f"<p>{e}</p>")
    return ("<html><body style='font-family:sans-serif;max-width:800px;margin:auto'>"
            + "\n".join(out) + "</body></html>")


def to_pdf(md):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(True, 15)
    for l in md.split("\n"):
        t = l.encode("latin-1", "replace").decode("latin-1")
        if l.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            t = t[3:]
        elif l.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            t = t[2:]
        else:
            pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, t or " ", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


with t5:
    st.subheader("Decision Receipt")
    st.plotly_chart(fig, width="stretch", key="receipt_fig")
    st.text_area("Request human review (optional): what should a reviewer check?", key="review_note")
    md = build_receipt()
    st.markdown(md)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Receipt (.html)", to_html(md), "decision_receipt.html", "text/html")
    d2.download_button("⬇ Receipt (.md)", md, "decision_receipt.md", "text/markdown")
    if HAS_PDF:
        d3.download_button("⬇ Receipt (.pdf)", to_pdf(md), "decision_receipt.pdf", "application/pdf")


# ---------------------------------------------------------------- validation
TRUE_W = {"gpa": 0.9, "test_score": 0.7, "family_income_k": 0.6, "essay_score": 0.5,
          "recommendation": 0.4, "extracurriculars": 0.3}


@st.cache_resource
def detection():
    smp = M["te"].sample(300, random_state=3)

    def flagged(df):
        return float(np.mean([any(sev == "High" for sev, _ in data_checks(x, False, False))
                              for x in df[FEATURES].to_dict("records")]))

    def preds(df):
        return M["model"].predict_proba(M["sc"].transform(df[FEATURES]))[:, 1] >= THRESHOLD

    zero = smp.assign(test_score=0.0)
    imputed = smp.assign(test_score=float(M["all"].test_score.median()))
    was = smp.pred.values == 1
    flip = float((~preds(zero)[was]).mean()) if was.any() else float("nan")
    r = np.random.default_rng(7)
    tr = M["tr"].copy()
    hit = (tr.region == "Rural") & (tr.y == 1) & (r.random(len(tr)) < 0.3)
    tr.loc[hit, "y"] = 0  # simulated bias: some rural positives wrongly labelled "no"
    bm = LogisticRegression(max_iter=1000).fit(M["sc"].transform(tr[FEATURES]), tr.y)
    bp = pd.Series(bm.predict_proba(M["sc"].transform(M["te"][FEATURES]))[:, 1] >= THRESHOLD, index=M["te"].index)
    br = bp.groupby(M["te"].region).mean()
    cr = M["te"].groupby("region").pred.mean()
    return dict(zero=flagged(zero), imputed=flagged(imputed), flip=flip, n=len(smp),
                clean=cr.min() / cr.max(), biased=br.min() / br.max(), biased_rates=br)


with t6:
    m = M["m"]
    st.subheader("Model quality")
    cvdf = pd.DataFrame({"Model": list(M["cv"]), "5-fold CV ROC-AUC (train)": list(M["cv"].values()),
                         "Test ROC-AUC": [M["test_auc"][n] for n in M["cv"]]})
    st.dataframe(cvdf.round(3), hide_index=True, width="stretch")
    st.caption(f"Selected by cross-validation (simpler model preferred unless clearly worse): **{M['best']}**.")
    q = st.columns(4)
    q[0].metric("Test accuracy", f"{m['accuracy']:.1%}", f"baseline {m['base_acc']:.1%}", delta_color="off")
    q[1].metric("Test ROC-AUC", f"{m['auc']:.3f}", "baseline 0.500", delta_color="off")
    q[2].metric("Brier score (lower=better)", f"{m['brier']:.3f}")
    q[3].metric("Selected model", M["best"])
    mp, fp = m["cal"]
    cal = go.Figure([go.Scatter(x=mp, y=fp, mode="lines+markers", name="Model"),
                     go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash"))])
    cal.update_layout(height=300, xaxis_title="Predicted probability", yaxis_title="Observed shortlist rate",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(cal, width="stretch", key="cal_fig")

    st.subheader("Detection test (injected problems)")
    D = detection()
    st.write(f"**Injected missing test score** (recorded as 0) in {D['n']} test applicants: the input check flagged "
             f"**{D['zero']:.0%}**; among applicants the model had shortlisted, **{D['flip']:.0%}** flipped to "
             f"'Not shortlisted'.")
    st.write(f"**Injected silent imputation** (missing test score replaced by the median): the input check flagged "
             f"**{D['imputed']:.0%}**. " + ("This is a MISS - silently imputed values cannot be detected by these rules."
                                          if D["imputed"] < 0.5 else ""))
    st.write(f"**Injected simulated bias** (30% of rural positive labels flipped to 'no' in training): selection-rate "
             f"ratio **{D['biased']:.2f}** vs **{D['clean']:.2f}** for the clean model -> "
             + ("flagged by the 0.80 audit rule." if D["biased"] < 0.8 else "NOT flagged by the 0.80 audit rule (a miss)."))

    st.subheader("Explanation check (SHAP vs the known synthetic rules)")
    gv = np.abs(raw).mean(0)
    ex = pd.DataFrame({"Feature": [NAMES[f] for f in FEATURES], "True rule weight": [TRUE_W[f] for f in FEATURES],
                       "Mean |SHAP|": gv})
    st.dataframe(ex.round(3), hide_index=True, width="stretch")
    rho = pd.Series(ex["True rule weight"].values).corr(pd.Series(gv), method="spearman")
    t3t = set(ex.nlargest(3, "True rule weight").Feature) == set(ex.nlargest(3, "Mean |SHAP|").Feature)
    st.write(f"Rank correlation (Spearman) = **{rho:.2f}**. Top-3 drivers {'match' if t3t else 'do NOT match'} the rules.")

    w1, w2 = st.columns(2)
    w1.success("**What this proves:** the prototype behaves as designed on controlled synthetic data.")
    w2.error("**What it does not prove:** real-world fairness, causation or discrimination.")
