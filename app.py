import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# ============================================================
# Utilities
# ============================================================

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def oversample_minority(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Simple random oversampling without external deps."""
    rng = np.random.default_rng(random_state)
    y = y.astype(int).copy()
    X = X.copy()

    cls_counts = y.value_counts()
    if len(cls_counts) != 2:
        return X, y

    maj = cls_counts.idxmax()
    minc = cls_counts.idxmin()
    n_maj = int(cls_counts.max())
    n_min = int(cls_counts.min())
    if n_min == 0 or n_min == n_maj:
        return X, y

    idx_min = y[y == minc].index.to_numpy()
    add_idx = rng.choice(idx_min, size=(n_maj - n_min), replace=True)
    X_os = pd.concat([X, X.loc[add_idx]], axis=0)
    y_os = pd.concat([y, y.loc[add_idx]], axis=0)
    return X_os, y_os

def plot_roc(y_true, proba, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    return fig

def make_quantile_grades(proba, n_bins=10):
    # Higher PD => worse grade (larger number)
    s = pd.Series(proba).rank(method="first")
    bins = pd.qcut(s, q=n_bins, labels=False, duplicates="drop") + 1
    return bins.astype(int)

def fit_stepwise_logit_forward_backward(
    X: pd.DataFrame,
    y: pd.Series,
    p_enter: float = 0.05,
    p_remove: float = 0.10,
    max_iter: int = 100
):
    """
    Forward selection with backward elimination using statsmodels Logit.
    Starts empty, iteratively adds the best variable with p < p_enter,
    then removes any variable with p > p_remove. Returns selected columns and fitted model.
    """
    y = y.astype(int)
    remaining = list(X.columns)
    selected: list[str] = []
    model = None

    def _fit(cols):
        Xc = sm.add_constant(X[cols], has_constant="add")
        # Use try/except for separation / singularities
        return sm.Logit(y, Xc).fit(disp=False, maxiter=max_iter)

    changed = True
    while changed:
        changed = False

        # -------- Forward step
        best_p = None
        best_col = None
        best_model = None
        for col in remaining:
            cols_try = selected + [col]
            try:
                m = _fit(cols_try)
                p = float(m.pvalues.get(col, 1.0))
            except Exception:
                continue
            if best_p is None or p < best_p:
                best_p = p
                best_col = col
                best_model = m

        if best_col is not None and best_p is not None and best_p < p_enter:
            selected.append(best_col)
            remaining.remove(best_col)
            model = best_model
            changed = True

        # -------- Backward step
        if selected:
            try:
                m_full = _fit(selected)
                # drop const
                pvals = m_full.pvalues.drop(labels=["const"], errors="ignore")
                worst_p = float(pvals.max()) if len(pvals) else 0.0
                if len(pvals) and worst_p > p_remove:
                    worst_col = str(pvals.idxmax())
                    selected.remove(worst_col)
                    remaining.append(worst_col)
                    model = _fit(selected) if selected else None
                    changed = True
                else:
                    model = m_full
            except Exception:
                # if refit fails, keep last model
                pass

    return model, selected

def sklearn_learning_curve_plot(estimator, X, y, cv_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator,
        X, y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=None,
        train_sizes=np.linspace(0.1, 1.0, 8)
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_sizes, train_scores.mean(axis=1))
    ax.plot(train_sizes, valid_scores.mean(axis=1))
    ax.set_xlabel("Training examples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Learning Curve (ROC-AUC)")
    ax.legend(["Train", "CV"], loc="best")
    return fig


# ============================================================
# App UI
# ============================================================

st.set_page_config(page_title="Credit Default Modeling", layout="wide")
st.title("부실율(0/1) 예측 Streamlit 앱")

tabs = st.tabs([
    "1) 데이터 탐색(EDA)",
    "2) 전처리 & 분할",
    "3) 모델링",
    "4) 성능평가",
    "5) PD 기반 고객세분화/부실율"
])

# ---------------------------
# 0) Upload
# ---------------------------
with st.sidebar:
    st.header("전역 설정")
    random_state = st.number_input("Random State", min_value=0, max_value=999999, value=42, step=1)
    test_size = st.slider("Test 비율", 0.1, 0.5, 0.2, 0.05)
    st.caption("Random state는 분할/모델 재현성을 위해 사용됩니다.")

uploaded = st.file_uploader("CSV 업로드", type=["csv"])
if uploaded is None:
    st.info("CSV 파일을 업로드하면 탭별 기능이 활성화됩니다.")
    st.stop()

# encoding fallback
raw = uploaded.read()
for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding=enc)
        break
    except Exception:
        df = None
if df is None:
    st.error("CSV 인코딩을 확인할 수 없습니다. utf-8/cp949/euc-kr 등을 확인해 주세요.")
    st.stop()

st.session_state.setdefault("df", df)

with st.sidebar:
    st.subheader("타깃 설정")
    target_col = st.selectbox("종속변수(0/1)", options=df.columns.tolist(), index=df.columns.get_loc("not.fully.paid") if "not.fully.paid" in df.columns else 0)
    st.caption("0/1 이진 변수만 지원합니다.")

# basic target validation
if target_col not in df.columns:
    st.error("타깃 컬럼이 데이터에 없습니다.")
    st.stop()

# ============================================================
# 1) EDA
# ============================================================
with tabs[0]:
    st.subheader("1) 데이터 탐색(EDA)")
    st.write("행/열:", df.shape)
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 결측치 요약")
        na = df.isna().mean().sort_values(ascending=False).to_frame("missing_ratio")
        st.dataframe(na.head(30), use_container_width=True)
    with col2:
        st.markdown("#### 타깃 분포")
        vc = df[target_col].value_counts(dropna=False)
        st.write(vc)
        if vc.sum() > 0:
            st.write("양성(1) 비율:", safe_div(int(vc.get(1, 0)), int(vc.sum())))

    st.markdown("#### 수치형 상관관계(참고)")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols = [c for c in num_cols if c != target_col]
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        st.dataframe(corr.round(3), use_container_width=True)
    else:
        st.info("상관관계를 계산할 수 있는 수치형 변수가 충분하지 않습니다.")

# ============================================================
# 2) Preprocess & Split
# ============================================================
with tabs[1]:
    st.subheader("2) 전처리 & 분할")
    st.caption("권장 흐름: (1) 전처리 → (2) Train/Test 분할 → (3) (선택) 변수선택/모델링")

    if target_col not in df.columns:
        st.stop()

    # Identify feature columns
    y_raw = df[target_col].astype(int)
    X_raw = df.drop(columns=[target_col])

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_raw.columns if c not in numeric_cols]

    st.markdown("#### 전처리 설정")
    c1, c2, c3 = st.columns(3)
    with c1:
        do_outlier = st.checkbox("수치형 이상치 제거(IQR)", value=True)
        iqr_k = st.slider("IQR k", 1.0, 3.0, 1.5, 0.1, disabled=not do_outlier)
    with c2:
        scale_numeric = st.checkbox("수치형 표준화(StandardScaler)", value=True)
    with c3:
        drop_na_target = st.checkbox("타깃 결측 제거", value=True)

    if st.button("전처리 실행"):
        dfx = df.copy()
        if drop_na_target:
            dfx = dfx.dropna(subset=[target_col])

        # outlier removal on numeric features only (row-wise)
        if do_outlier and len(numeric_cols) > 0:
            tmp = dfx[numeric_cols]
            q1 = tmp.quantile(0.25)
            q3 = tmp.quantile(0.75)
            iqr = (q3 - q1).replace(0, np.nan)
            lower = q1 - iqr_k * iqr
            upper = q3 + iqr_k * iqr
            mask = np.ones(len(dfx), dtype=bool)
            for c in numeric_cols:
                if pd.isna(iqr[c]):
                    continue
                mask &= (dfx[c] >= lower[c]) & (dfx[c] <= upper[c])
            dfx = dfx.loc[mask].copy()

        y0 = dfx[target_col].astype(int)
        X0 = dfx.drop(columns=[target_col])

        num_cols2 = X0.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols2 = [c for c in X0.columns if c not in num_cols2]

        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if scale_numeric else "passthrough"),
        ])
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols2),
                ("cat", cat_pipe, cat_cols2),
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        Xp = preprocessor.fit_transform(X0)
        feat_names = preprocessor.get_feature_names_out()
        Xp = pd.DataFrame(Xp, columns=feat_names, index=X0.index)

        st.session_state["preprocessor"] = preprocessor
        st.session_state["X_processed"] = Xp
        st.session_state["y_processed"] = y0
        st.success("전처리가 완료되었습니다.")

    if "X_processed" in st.session_state:
        Xp = st.session_state["X_processed"]
        yp = st.session_state["y_processed"]
        st.write(f"전처리 후 X shape: {Xp.shape} / y length: {len(yp)}")

        if st.button("Train/Test 분할 저장"):
            X_train, X_test, y_train, y_test = train_test_split(
                Xp, yp,
                test_size=float(test_size),
                random_state=int(random_state),
                stratify=yp if yp.nunique() == 2 else None
            )
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.success("분할이 완료되었습니다.")

        if all(k in st.session_state for k in ["X_train","X_test","y_train","y_test"]):
            st.write("Train shape:", st.session_state["X_train"].shape, "/ Test shape:", st.session_state["X_test"].shape)

# ============================================================
# 3) Modeling
# ============================================================
with tabs[2]:
    st.subheader("3) 모델링")

    required = ["X_train", "X_test", "y_train", "y_test"]
    missing = [k for k in required if k not in st.session_state]
    if missing:
        st.warning("먼저 [전처리 & 분할] 탭에서 전처리 및 Train/Test 분할을 완료하세요.")
        st.stop()

    X_train = st.session_state["X_train"]
    X_test = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test = st.session_state["y_test"]

    st.markdown("#### 모델 선택")
    model_type = st.selectbox("모델", ["Logistic Regression", "Decision Tree", "MLP(신경망)"])

    st.markdown("#### 데이터 불균형 처리")
    imb_mode = st.selectbox("방법", ["없음", "Class Weight (balanced)", "Random Oversampling"], index=1)

    st.markdown("#### (선택) Stepwise 변수선택 (Forward + Backward)")
    use_stepwise = (model_type == "Logistic Regression") and st.checkbox("Stepwise 적용", value=True)
    if use_stepwise:
        c1, c2, c3 = st.columns(3)
        with c1:
            p_enter = st.number_input("진입(p_enter)", min_value=0.001, max_value=0.5, value=0.05, step=0.005)
        with c2:
            p_remove = st.number_input("제거(p_remove)", min_value=0.001, max_value=0.8, value=0.10, step=0.01)
        with c3:
            max_iter = st.number_input("max_iter(statsmodels)", min_value=20, max_value=500, value=100, step=10)

    st.divider()

    if st.button("모델 학습 및 예측확률 생성"):
        X_tr = X_train.copy()
        y_tr = y_train.copy()

        # imbalance handling (training only)
        class_weight = None
        if imb_mode == "Class Weight (balanced)":
            class_weight = "balanced"
        elif imb_mode == "Random Oversampling":
            X_tr, y_tr = oversample_minority(X_tr, y_tr, random_state=int(random_state))

        selected_cols = list(X_tr.columns)

        # stepwise selection on (possibly resampled) training set
        if use_stepwise:
            with st.spinner("Stepwise Logit 변수선택 수행 중..."):
                sm_model, selected_cols = fit_stepwise_logit_forward_backward(
                    X_tr, y_tr,
                    p_enter=float(p_enter),
                    p_remove=float(p_remove),
                    max_iter=int(max_iter)
                )
            if len(selected_cols) == 0:
                st.error("Stepwise 결과 선택된 변수가 없습니다. p_enter/p_remove를 완화하거나 전처리를 점검하세요.")
                st.stop()
            st.session_state["sm_stepwise_model"] = sm_model
            st.session_state["selected_cols"] = selected_cols
            st.success(f"선택 변수 수: {len(selected_cols)}")

        # Fit chosen sklearn model (for prediction, logloss, learning curve)
        if model_type == "Logistic Regression":
            est = LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight=class_weight,
                random_state=int(random_state)
            )
        elif model_type == "Decision Tree":
            est = DecisionTreeClassifier(
                random_state=int(random_state),
                class_weight=("balanced" if class_weight == "balanced" else None)
            )
        else:
            # MLP: no class_weight; oversampling recommended
            est = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.0005,
                max_iter=400,
                random_state=int(random_state)
            )

        est.fit(X_tr[selected_cols], y_tr)
        proba_test = est.predict_proba(X_test[selected_cols])[:, 1]

        st.session_state["estimator"] = est
        st.session_state["model_type"] = model_type
        st.session_state["imb_mode"] = imb_mode
        st.session_state["proba_test"] = proba_test
        st.session_state["selected_cols"] = selected_cols
        st.success("모델 학습 및 예측확률(PD) 생성이 완료되었습니다.")

    # Show coefficients / p-values when available
    if model_type == "Logistic Regression" and "sm_stepwise_model" in st.session_state and st.session_state.get("selected_cols"):
        st.markdown("#### Stepwise(Logit) 결과 요약 (statsmodels)")
        sm_model = st.session_state["sm_stepwise_model"]
        if sm_model is not None:
            try:
                summ = sm_model.summary2().tables[1]
                st.dataframe(summ, use_container_width=True)
            except Exception:
                st.info("statsmodels 요약을 표시할 수 없습니다.")

# ============================================================
# 4) Evaluation
# ============================================================
with tabs[3]:
    st.subheader("4) 성능평가")

    required = ["y_test", "proba_test", "X_train", "y_train", "X_test"]
    missing = [k for k in required if k not in st.session_state]
    if missing:
        st.warning("먼저 [모델링] 탭에서 모델을 학습하세요.")
        st.stop()

    y_test = st.session_state["y_test"]
    proba_test = st.session_state["proba_test"]
    y_pred = (proba_test >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, proba_test)
    ll = log_loss(y_test, proba_test, labels=[0,1])

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.4f}")
    c1.metric("ROC-AUC", f"{auc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c2.metric("Recall", f"{rec:.4f}")
    c3.metric("F1-score", f"{f1:.4f}")
    c3.metric("Cross-Entropy (Log Loss)", f"{ll:.4f}")

    st.divider()

    # Confusion matrix
    st.markdown("#### Confusion Matrix (threshold=0.5)")
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df, use_container_width=True)

    # ROC curve
    st.markdown("#### ROC Curve")
    st.pyplot(plot_roc(y_test, proba_test, title="ROC Curve"))

    st.divider()

    # Learning curve
    st.markdown("#### Learning Curve")
    if "estimator" not in st.session_state:
        st.info("학습곡선을 위해서는 sklearn estimator가 필요합니다.")
    else:
        est = st.session_state["estimator"]
        selected_cols = st.session_state.get("selected_cols", list(st.session_state["X_train"].columns))
        X_train = st.session_state["X_train"][selected_cols]
        y_train = st.session_state["y_train"]

        try:
            fig = sklearn_learning_curve_plot(est, X_train, y_train, cv_splits=5, random_state=int(random_state))
            st.pyplot(fig)
            st.caption("스코어는 ROC-AUC 기준입니다. Train/CV 간 갭이 크면 과적합 가능성이 있습니다.")
        except Exception as e:
            st.info(f"Learning curve를 계산할 수 없습니다: {e}")

# ============================================================
# 5) Segmentation
# ============================================================
with tabs[4]:
    st.subheader("5) PD 기반 고객세분화/부실율")

    required = ["y_test", "proba_test"]
    missing = [k for k in required if k not in st.session_state]
    if missing:
        st.warning("먼저 [모델링] 탭에서 예측확률(PD)을 생성하세요.")
        st.stop()

    y_test = st.session_state["y_test"]
    proba_test = st.session_state["proba_test"]
    st.write(f"Test 샘플 수: {len(y_test)}")

    st.markdown("### PD Segmentation 설정")
    n_bins = st.slider("등급 수 (Grade 개수)", 5, 20, 10, 1)

    grades = make_quantile_grades(proba_test, n_bins=n_bins)
    seg = pd.DataFrame({
        "PD": proba_test,
        "Grade": grades,
        "Default(1)": y_test.values
    })

    summary = (
        seg.groupby("Grade")
        .agg(n=("PD", "size"), avg_pd=("PD", "mean"), default_rate=("Default(1)", "mean"))
        .reset_index()
        .sort_values("Grade")
    )
    st.dataframe(summary, use_container_width=True)

    st.markdown("#### Grade별 PD/부실율 시각화")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(summary["Grade"], summary["avg_pd"])
    ax.plot(summary["Grade"], summary["default_rate"])
    ax.set_xlabel("Grade")
    ax.set_ylabel("Rate")
    ax.set_title("Avg PD vs Default Rate by Grade")
    ax.legend(["Avg PD", "Default rate"], loc="best")
    st.pyplot(fig)
