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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import shap
    HAS_SHAP = True
except Exception:  # fallback keeps the demo alive if shap fails to install
    HAS_SHAP = False

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
    Xtr = sc.transform(tr[FEATURES])
    model = LogisticRegression(max_iter=1000).fit(Xtr, tr.y)
    te = te.copy()
    te["prob"] = model.predict_proba(sc.transform(te[FEATURES]))[:, 1]
    te["pred"] = (te.prob >= THRESHOLD).astype(int)
    expl = shap.LinearExplainer(model, Xtr) if HAS_SHAP else None
    allp = d.copy()
    allp["prob"] = model.predict_proba(sc.transform(d[FEATURES]))[:, 1]
    allp["pred"] = (allp.prob >= THRESHOLD).astype(int)
    metrics = dict(accuracy=accuracy_score(te.y, te.pred), precision=precision_score(te.y, te.pred),
                   recall=recall_score(te.y, te.pred), f1=f1_score(te.y, te.pred),
                   cm=confusion_matrix(te.y, te.pred))
    base = float(model.decision_function(Xtr).mean())
    lo, hi = tr[FEATURES].quantile(0.01), tr[FEATURES].quantile(0.99)
    return dict(sc=sc, model=model, te=te, expl=expl, all=allp, m=metrics, base=base, lo=lo, hi=hi, ntrain=len(tr))


M = train()


def predict(row):
    xs = M["sc"].transform(pd.DataFrame([row])[FEATURES])
    return float(M["model"].predict_proba(xs)[0, 1]), xs


def contribs(xs):
    """SHAP values (log-odds) for one scaled row; falls back to coef*x (identical for a linear model)."""
    if M["expl"] is not None:
        try:
            return np.array(M["expl"].shap_values(xs)).reshape(-1)
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
    return f"{NAMES[f]} = {v:.1f} {'pushed the score UP' if c > 0 else 'pushed the score DOWN'} ({c:+.2f} log-odds)"


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

t1, t2, t3, t4, t5 = st.tabs(["1 Explain", "2 Check data", "3 Compare", "4 Model & groups", "5 Decision Receipt"])

# ---------------------------------------------------------------- explain
with t1:
    st.subheader("Why did the model predict this? (local SHAP)")
    order = np.argsort(np.abs(sv))
    fig = go.Figure(go.Bar(x=sv[order], y=[NAMES[FEATURES[i]] for i in order], orientation="h",
                           marker_color=["#2e7d32" if v > 0 else "#c62828" for v in sv[order]]))
    fig.update_layout(height=320, xaxis_title="Contribution to log-odds of shortlisting",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Baseline log-odds {M['base']:+.2f} + contributions {sv.sum():+.2f} = "
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
         f"- Model prediction: {dec}", f"- Prediction score: {prob:.1%} (threshold {THRESHOLD:.0%})",
         f"- Model: logistic regression trained on {M['ntrain']} synthetic applicants", "",
         "## 2. Why did it happen?"] + [f"- {t}" for t in top] + [
         "- These are model contributions (SHAP), not proof of causation.", "",
         "## 3. What should be checked?"]
    L += [f"- [{s}] {m_}" for s, m_ in checks] or ["- No automated input issues detected (does not guarantee correctness)."]
    if S.test_missing or S.income_missing:
        L.append(f"- With missing values resolved the output would be: {decision(p_res)} ({p_res:.1%})")
    L += ["", "## 4. Is there a warning signal?",
          f"- Test accuracy {M['m']['accuracy']:.1%}, precision {M['m']['precision']:.1%}, "
          f"recall {M['m']['recall']:.1%}, F1 {M['m']['f1']:.1%}",
          f"- Selection-rate ratio across regions: {di:.2f} (lower group: {low_g})"
          + (" - below 0.80, investigate" if di < 0.8 else " - above 0.80"),
          "- Closest synthetic applicants with a different outcome:"]
    for _, r_ in neighbor.iterrows():
        L.append(f"- GPA {r_.gpa:.2f}, test {r_.test_score:.0f}, essay {r_.essay_score:.1f}, "
                 f"income {r_.family_income_k:.0f}k -> {decision(r_.prob)} ({r_.prob:.1%})")
    L += ["", "## 5. What happens next? Questions for human review",
          "- Is the flagged information correct? Verify against source documents." if checks
          else "- Confirm the inputs against source documents.",
          "- Would a human reviewer reach the same outcome after correcting the record?",
          "- Does the selection-rate gap between groups have a legitimate explanation?",
          "- Can the applicant see this explanation and request a review?", "",
          "## Limitations",
          "- Feature importance does not prove causation.",
          "- A changed prediction after editing an input does not by itself prove discrimination.",
          "- Group differences are signals for investigation, not verified findings.",
          "- Performance on synthetic data does not show validity for real decisions.",
          "- This prototype must not be used for actual eligibility or scholarship decisions."]
    return "\n".join(L)


def to_html(md):
    out = []
    for l in md.split("\n"):
        e = html.escape(l.strip())
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


with t5:
    st.subheader("Decision Receipt")
    md = build_receipt()
    st.markdown(md)
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download receipt (.md)", md, "decision_receipt.md", "text/markdown")
    d2.download_button("⬇ Download receipt (.html)", to_html(md), "decision_receipt.html", "text/html")
