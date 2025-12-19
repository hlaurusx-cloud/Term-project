# app.py
# ------------------------------------------------------------
# 개인신용평가(상환/부실 예측) Streamlit 전체 앱
# - Data Load
# - Target Labeling (binary)
# - Preprocess (missing, one-hot, scaling)
# - Feature Selection (Stepwise backward for Logit: p-value 기준)
# - Models: Logit (statsmodels), Neural Net (sklearn MLPClassifier)
# - Evaluation: ROC-AUC, Acc, Precision, Recall, F1, Confusion Matrix, ROC plot
# - Segmentation: PD quantiles -> grades, observed default rate
# ------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from sklearn.neural_network import MLPClassifier

import statsmodels.api as sm


# -----------------------------
# Utility
# -----------------------------
def safe_read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    # try utf-8 then cp949
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    # fallback
    return pd.read_csv(io.BytesIO(raw), encoding_errors="ignore")


def make_binary_target(df: pd.DataFrame, target_col: str, positive_classes: list) -> pd.Series:
    """
    Returns y (0/1). positive_classes -> 1 (부실), else -> 0
    """
    y = df[target_col].astype(str).isin([str(x) for x in positive_classes]).astype(int)
    return y


def split_numeric_categorical(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, num_cols, cat_cols


def preprocess_fit_transform(
    df: pd.DataFrame,
    feature_cols: list,
    scaler_on: bool = True,
    drop_high_cardinality: bool = True,
    high_cardinality_threshold: int = 100,
):
    """
    Fit preprocessing on train only. Here we do a simple approach:
    - Split numeric/categorical
    - Missing: numeric median, categorical 'Unknown'
    - Optionally drop very high-cardinality categorical columns
    - One-hot encode categorical
    - Standardize numeric (optional)
    Returns:
        X_proc (pd.DataFrame), preprocess_artifacts dict
    """
    X, num_cols, cat_cols = split_numeric_categorical(df, feature_cols)

    # high-cardinality drop (optional)
    dropped_cat = []
    if drop_high_cardinality and len(cat_cols) > 0:
        for c in list(cat_cols):
            nunique = X[c].nunique(dropna=True)
            if nunique > high_cardinality_threshold:
                dropped_cat.append(c)
        cat_cols = [c for c in cat_cols if c not in dropped_cat]
        X = X[[c for c in X.columns if c not in dropped_cat]]

    # missing handling
    for c in num_cols:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        X[c] = X[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].astype("object").fillna("Unknown")

    # one-hot
    X_cat = pd.get_dummies(X[cat_cols], drop_first=False) if len(cat_cols) else None
    X_num = X[num_cols] if len(num_cols) else None

    if X_num is None and X_cat is None:
        raise ValueError("선택된 feature가 없습니다. X(특징 변수) 선택을 다시 확인하세요.")

    if X_num is None:
        X_proc = X_cat.copy()
        scaler = None
        num_cols_used = []
    elif X_cat is None:
        X_proc = X_num.copy()
        num_cols_used = num_cols
        scaler = None
        if scaler_on:
            scaler = StandardScaler()
            X_proc[num_cols_used] = scaler.fit_transform(X_proc[num_cols_used])
    else:
        X_proc = pd.concat([X_num, X_cat], axis=1)
        num_cols_used = num_cols
        scaler = None
        if scaler_on:
            scaler = StandardScaler()
            X_proc[num_cols_used] = scaler.fit_transform(X_proc[num_cols_used])

    artifacts = {
        "num_cols": num_cols_used,
        "cat_cols": cat_cols,
        "dropped_cat": dropped_cat,
        "onehot_columns": X_proc.columns.tolist(),
        "scaler": scaler,
    }
    return X_proc, artifacts


def preprocess_transform(df: pd.DataFrame, feature_cols: list, artifacts: dict):
    """
    Transform using trained artifacts:
    - same drop columns
    - same missing strategy (median for numeric is NOT stored here; we reuse median from current df, acceptable for demo.
      For production, store train medians and reuse them.)
    - one-hot with fixed columns, align
    - scaling with stored scaler for numeric cols
    """
    X = df[feature_cols].copy()
    # drop high-cardinality columns
    for c in artifacts.get("dropped_cat", []):
        if c in X.columns:
            X = X.drop(columns=[c])

    # numeric/categorical based on current dtypes and stored cat cols
    stored_cat_cols = artifacts.get("cat_cols", [])
    # Determine numeric columns as in current frame
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # Keep categorical as stored (if present)
    cat_cols = [c for c in stored_cat_cols if c in X.columns]
    # Fill missing
    for c in num_cols:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        X[c] = X[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].astype("object").fillna("Unknown")

    X_cat = pd.get_dummies(X[cat_cols], drop_first=False) if len(cat_cols) else None
    X_num = X[num_cols] if len(num_cols) else None

    if X_num is None and X_cat is None:
        raise ValueError("변환할 feature가 없습니다.")

    if X_num is None:
        X_proc = X_cat.copy()
    elif X_cat is None:
        X_proc = X_num.copy()
    else:
        X_proc = pd.concat([X_num, X_cat], axis=1)

    # align columns to onehot_columns
    cols = artifacts["onehot_columns"]
    X_proc = X_proc.reindex(columns=cols, fill_value=0.0)

    # scaling numeric
    scaler = artifacts.get("scaler", None)
    num_cols_used = artifacts.get("num_cols", [])
    if scaler is not None and len(num_cols_used) > 0:
        X_proc[num_cols_used] = scaler.transform(X_proc[num_cols_used])

    return X_proc


def stepwise_backward_logit(X_train: pd.DataFrame, y_train: pd.Series, p_threshold: float = 0.05, max_iter: int = 100):
    """
    Backward elimination using statsmodels Logit p-values (Wald).
    Adds constant.
    Handles common issues by removing problematic columns:
      - constant columns
      - singular matrix (multicollinearity/perfect separation) -> remove the worst p-value variable iteratively
    Returns: fitted_model, selected_cols, removed_log
    """
    removed_log = []
    X = X_train.copy()

    # drop constant columns
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        removed_log.extend([(c, "constant_col") for c in constant_cols])

    cols = X.columns.tolist()
    y_num = pd.Series(y_train).astype(int)

    for _ in range(max_iter):
        X_const = sm.add_constant(X[cols], has_constant="add")
        try:
            model = sm.Logit(y_num, X_const).fit(disp=False, maxiter=200)
        except Exception as e:
            # try to remove the most problematic variable by high VIF proxy: remove one with highest abs correlation sum
            if len(cols) <= 1:
                raise RuntimeError(f"Logit 학습 실패(변수 너무 적음). 마지막 오류: {e}")
            # heuristic: remove column with highest NaN or variance issues or correlation sum
            corr = X[cols].corr().abs()
            corr_sum = corr.sum().sort_values(ascending=False)
            drop_candidate = corr_sum.index[0]
            cols.remove(drop_candidate)
            removed_log.append((drop_candidate, f"fit_error:{type(e).__name__}"))
            continue

        pvalues = model.pvalues.drop("const", errors="ignore")
        if pvalues.empty:
            break

        worst_p = pvalues.max()
        worst_var = pvalues.idxmax()

        if worst_p <= p_threshold:
            return model, cols, removed_log

        cols.remove(worst_var)
        removed_log.append((worst_var, f"p={worst_p:.4f}"))

        if len(cols) == 0:
            break

    # final fit with remaining cols
    if len(cols) == 0:
        raise RuntimeError("Stepwise 결과 선택된 변수가 0개입니다. p_threshold를 완화하거나 feature를 재선택하세요.")
    X_const = sm.add_constant(X[cols], has_constant="add")
    model = sm.Logit(y_num, X_const).fit(disp=False, maxiter=200)
    return model, cols, removed_log


def eval_binary_from_proba(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, proba),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ConfusionMatrix": confusion_matrix(y_true, pred),
        "pred": pred
    }


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


def make_risk_grades(proba: np.ndarray, n_bins: int = 5):
    """
    Quantile-based binning. Higher proba => higher risk.
    Returns grade labels (A lowest risk ... )
    """
    s = pd.Series(proba)
    # qcut can fail when duplicates; use rank-based
    r = s.rank(method="average")
    grade_idx = pd.qcut(r, q=n_bins, labels=False, duplicates="drop")  # 0..n_bins-1
    # if duplicates drop reduces bins, handle
    actual_bins = int(pd.Series(grade_idx).nunique())
    labels = [chr(ord("A") + i) for i in range(actual_bins)]
    grade = pd.Series(grade_idx).map(lambda i: labels[int(i)] if pd.notna(i) else labels[-1])
    return grade, labels


def segment_table(y_true, proba, n_bins=5):
    grade, labels = make_risk_grades(proba, n_bins=n_bins)

    df = pd.DataFrame({
        "PD": proba,
        "Y": np.array(y_true).astype(int),
        "Grade": grade
    })

    agg = df.groupby("Grade").agg(
        Customers=("Y", "count"),
        Avg_PD=("PD", "mean"),
        Default_Rate=("Y", "mean")
    ).reset_index()

    # Ensure ordering A.. (A lowest risk)
    agg["GradeOrder"] = agg["Grade"].apply(lambda x: ord(x) - ord("A"))
    agg = agg.sort_values("GradeOrder").drop(columns=["GradeOrder"])

    return agg, df, labels


def plot_default_rate_by_grade(seg_agg: pd.DataFrame, title="Default Rate by Risk Grade"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(seg_agg["Grade"], seg_agg["Default_Rate"])
    ax.set_xlabel("Risk Grade (A=Low → High)")
    ax.set_ylabel("Observed Default Rate")
    ax.set_title(title)
    return fig


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="개인신용평가(상환예측) - Logit + Neural Net", layout="wide")
st.title("개인신용평가(상환예측) Streamlit: Logit + 신경망(MLP) + 고객세분화/부실율")

# Session states
if "df" not in st.session_state:
    st.session_state.df = None
if "data_ready" not in st.session_state:
    st.session_state.data_ready = False
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None
if "splits" not in st.session_state:
    st.session_state.splits = {}
if "models" not in st.session_state:
    st.session_state.models = {}
if "proba" not in st.session_state:
    st.session_state.proba = {}  # model_name -> proba_test
if "selected_cols_logit" not in st.session_state:
    st.session_state.selected_cols_logit = None


tabs = st.tabs([
    "1) 데이터/타깃 설정",
    "2) 전처리 & 데이터 분할",
    "3) Feature Selection (Stepwise-Logit)",
    "4) 모델 학습 (Logit / 신경망)",
    "5) 성능 평가/비교",
    "6) 고객세분화/부실율/전략"
])

# -----------------------------
# Tab 1: Data + Target
# -----------------------------
with tabs[0]:
    st.subheader("1) 데이터 로드 및 타깃(부실) 라벨 정의")

    uploaded = st.file_uploader("CSV 업로드", type=["csv"])
    if uploaded is not None:
        df = safe_read_csv(uploaded)
        st.session_state.df = df
        st.success(f"데이터 로드 완료: {df.shape[0]:,} rows x {df.shape[1]:,} cols")

    df = st.session_state.df
    if df is None:
        st.info("먼저 CSV 파일을 업로드하세요.")
    else:
        st.write("미리보기(상위 20행)")
        st.dataframe(df.head(20), use_container_width=True)

        target_col = st.selectbox("타깃 컬럼 선택 (예: loan_status)", options=df.columns.tolist())
        st.caption("부실(1)로 볼 범주값들을 선택하세요. 나머지는 정상(0) 처리합니다.")
        unique_vals = sorted(df[target_col].astype(str).unique().tolist())
        positive_classes = st.multiselect("부실(1) 클래스 선택", options=unique_vals)

        # Feature columns
        default_features = [c for c in df.columns if c != target_col]
        feature_cols = st.multiselect("특징 변수(X) 선택", options=default_features, default=default_features)

        # Save selections
        if st.button("타깃/특징 설정 저장"):
            if len(positive_classes) == 0:
                st.error("부실(1) 클래스가 비어있습니다. 최소 1개 이상 선택하세요.")
            elif len(feature_cols) == 0:
                st.error("특징 변수(X)를 최소 1개 이상 선택하세요.")
            else:
                y = make_binary_target(df, target_col, positive_classes)
                st.session_state.splits["target_col"] = target_col
                st.session_state.splits["positive_classes"] = positive_classes
                st.session_state.splits["feature_cols"] = feature_cols
                st.session_state.splits["y_full"] = y
                st.session_state.data_ready = True
                st.success("타깃/특징 설정 저장 완료")

        if st.session_state.data_ready:
            y = st.session_state.splits["y_full"]
            st.write("타깃 분포(부실=1 기준)")
            st.write(y.value_counts(dropna=False).rename_axis("Y").to_frame("count"))

# -----------------------------
# Tab 2: Preprocess + split
# -----------------------------
with tabs[1]:
    st.subheader("2) 전처리 및 Train/Test 분할")

    if not st.session_state.data_ready:
        st.info("먼저 [1) 데이터/타깃 설정]에서 설정을 저장하세요.")
    else:
        df = st.session_state.df
        feature_cols = st.session_state.splits["feature_cols"]
        y = st.session_state.splits["y_full"]

        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test 비율", 0.1, 0.5, 0.2, 0.05)
        with col2:
            random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1)
        with col3:
            stratify_on = st.checkbox("Stratify(Y) 적용", value=True)

        st.markdown("**전처리 옵션**")
        c1, c2, c3 = st.columns(3)
        with c1:
            scaler_on = st.checkbox("수치형 표준화(StandardScaler) 적용 (신경망 권장)", value=True)
        with c2:
            drop_hc = st.checkbox("고카디널리티 범주형 컬럼 제거", value=True)
        with c3:
            hc_th = st.number_input("고카디널리티 기준(nunique)", min_value=20, max_value=5000, value=100, step=10)

        if st.button("전처리 + 분할 실행"):
            # split first (fit preprocessing on train only)
            strat = y if stratify_on else None
            df_train, df_test, y_train, y_test = train_test_split(
                df, y, test_size=test_size, random_state=int(random_state), stratify=strat
            )

            # fit preprocess on train
            X_train_proc, artifacts = preprocess_fit_transform(
                df_train, feature_cols,
                scaler_on=scaler_on,
                drop_high_cardinality=drop_hc,
                high_cardinality_threshold=int(hc_th)
            )
            # transform test
            X_test_proc = preprocess_transform(df_test, feature_cols, artifacts)

            st.session_state.artifacts = artifacts
            st.session_state.splits.update({
                "df_train": df_train,
                "df_test": df_test,
                "X_train": X_train_proc,
                "X_test": X_test_proc,
                "y_train": y_train.astype(int).values,
                "y_test": y_test.astype(int).values,
            })

            st.success("전처리 및 분할 완료")
            st.write("X_train shape:", X_train_proc.shape, " / X_test shape:", X_test_proc.shape)
            if artifacts["dropped_cat"]:
                st.warning(f"제거된 고카디널리티 범주형: {artifacts['dropped_cat']}")

        if "X_train" in st.session_state.splits:
            st.write("전처리된 X_train 미리보기(상위 10행)")
            st.dataframe(st.session_state.splits["X_train"].head(10), use_container_width=True)

# -----------------------------
# Tab 3: Stepwise for Logit
# -----------------------------
with tabs[2]:
    st.subheader("3) Feature Selection - Stepwise Backward(Logit, p-value 기준)")

    if "X_train" not in st.session_state.splits:
        st.info("먼저 [2) 전처리 & 데이터 분할]을 완료하세요.")
    else:
        X_train = st.session_state.splits["X_train"]
        y_train = st.session_state.splits["y_train"]

        col1, col2 = st.columns(2)
        with col1:
            p_threshold = st.slider("제거 기준 p-value (클수록 더 많이 제거)", 0.01, 0.30, 0.05, 0.01)
        with col2:
            max_iter = st.number_input("최대 반복", min_value=10, max_value=500, value=100, step=10)

        selection_mode = st.radio(
            "신경망 학습에 사용할 변수셋",
            ["(권장) Logit Stepwise 선택 변수와 동일", "전처리된 전체 변수 사용"],
            index=0
        )

        if st.button("Stepwise 실행 (Logit)"):
            try:
                model, selected_cols, removed_log = stepwise_backward_logit(
                    X_train, y_train, p_threshold=float(p_threshold), max_iter=int(max_iter)
                )
                st.session_state.models["logit_stepwise"] = model
                st.session_state.selected_cols_logit = selected_cols
                st.session_state.splits["nn_use_selected"] = (selection_mode.startswith("(권장)"))

                st.success(f"Stepwise 완료: 선택 변수 {len(selected_cols):,}개")
                st.write("선택된 변수(상위 200개까지 표시)")
                st.code("\n".join(selected_cols[:200]))

                if removed_log:
                    st.write("제거 로그(상위 200개까지)")
                    st.dataframe(pd.DataFrame(removed_log, columns=["removed_var", "reason"]).head(200), use_container_width=True)

                st.write("Logit 요약(일부)")
                st.text(model.summary().as_text()[:3000])

            except Exception as e:
                st.error(f"Stepwise/Logit 실행 실패: {e}")
                st.info("대응 팁: (1) p_threshold를 높여보세요 (예: 0.10~0.20). (2) 너무 많은 변수가 있으면 고카디널리티 제거/상관 높은 변수 제거를 강화하세요.")

# -----------------------------
# Tab 4: Train models
# -----------------------------
with tabs[3]:
    st.subheader("4) 모델 학습 - Logit / 신경망(MLP)")

    if "X_train" not in st.session_state.splits:
        st.info("먼저 [2) 전처리 & 데이터 분할]을 완료하세요.")
    else:
        X_train_all = st.session_state.splits["X_train"]
        X_test_all = st.session_state.splits["X_test"]
        y_train = st.session_state.splits["y_train"]
        y_test = st.session_state.splits["y_test"]

        # Decide feature set for models
        selected_cols = st.session_state.selected_cols_logit
        use_selected_for_nn = st.session_state.splits.get("nn_use_selected", True)

        model_choices = st.multiselect(
            "학습할 모델 선택",
            options=["Logit(전체변수)", "Logit(Stepwise)", "신경망 MLP"],
            default=["Logit(Stepwise)", "신경망 MLP"] if selected_cols is not None else ["Logit(전체변수)", "신경망 MLP"]
        )

        # Neural net params
        st.markdown("**신경망(MLP) 하이퍼파라미터**")
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            h1 = st.number_input("Hidden1", min_value=8, max_value=512, value=64, step=8)
        with n2:
            h2 = st.number_input("Hidden2 (0이면 1층)", min_value=0, max_value=512, value=32, step=8)
        with n3:
            alpha = st.number_input("L2(alpha)", min_value=0.0, max_value=0.01, value=0.0001, step=0.0001, format="%.4f")
        with n4:
            max_iter = st.number_input("MLP max_iter", min_value=50, max_value=2000, value=300, step=50)

        if st.button("모델 학습 실행"):
            # Logit full variables
            if "Logit(전체변수)" in model_choices:
                Xc_train = sm.add_constant(X_train_all, has_constant="add")
                Xc_test = sm.add_constant(X_test_all, has_constant="add")
                try:
                    logit_full = sm.Logit(y_train, Xc_train).fit(disp=False, maxiter=200)
                    st.session_state.models["logit_full"] = logit_full
                    proba = logit_full.predict(Xc_test)
                    st.session_state.proba["logit_full"] = np.array(proba, dtype=float)
                    st.success("Logit(전체변수) 학습 완료")
                except Exception as e:
                    st.error(f"Logit(전체변수) 학습 실패: {e}")

            # Logit stepwise variables
            if "Logit(Stepwise)" in model_choices:
                if selected_cols is None:
                    st.error("Stepwise 선택 변수가 없습니다. [3) Feature Selection]을 먼저 실행하세요.")
                else:
                    X_train_sel = X_train_all[selected_cols]
                    X_test_sel = X_test_all[selected_cols]
                    Xc_train = sm.add_constant(X_train_sel, has_constant="add")
                    Xc_test = sm.add_constant(X_test_sel, has_constant="add")
                    try:
                        logit_sw = sm.Logit(y_train, Xc_train).fit(disp=False, maxiter=200)
                        st.session_state.models["logit_stepwise_refit"] = logit_sw
                        proba = logit_sw.predict(Xc_test)
                        st.session_state.proba["logit_stepwise"] = np.array(proba, dtype=float)
                        st.success("Logit(Stepwise) 학습 완료")
                    except Exception as e:
                        st.error(f"Logit(Stepwise) 학습 실패: {e}")

            # Neural Net MLP
            if "신경망 MLP" in model_choices:
                # choose feature set
                if selected_cols is not None and use_selected_for_nn:
                    X_train_nn = X_train_all[selected_cols]
                    X_test_nn = X_test_all[selected_cols]
                    st.info("신경망은 Logit Stepwise 선택 변수를 사용합니다(공정 비교).")
                else:
                    X_train_nn = X_train_all
                    X_test_nn = X_test_all
                    st.info("신경망은 전처리된 전체 변수를 사용합니다(성능 우선).")

                hidden = (int(h1),) if int(h2) == 0 else (int(h1), int(h2))

                try:
                    mlp = MLPClassifier(
                        hidden_layer_sizes=hidden,
                        activation="relu",
                        solver="adam",
                        alpha=float(alpha),
                        max_iter=int(max_iter),
                        random_state=42
                    )
                    mlp.fit(X_train_nn, y_train)
                    st.session_state.models["nn_mlp"] = mlp
                    proba = mlp.predict_proba(X_test_nn)[:, 1]
                    st.session_state.proba["nn_mlp"] = np.array(proba, dtype=float)
                    st.success("신경망(MLP) 학습 완료")
                except Exception as e:
                    st.error(f"신경망(MLP) 학습 실패: {e}")

# -----------------------------
# Tab 5: Evaluation & comparison
# -----------------------------
with tabs[4]:
    st.subheader("5) 성능 평가 및 모델 비교")

    if "y_test" not in st.session_state.splits:
        st.info("먼저 [2) 전처리 & 데이터 분할]을 완료하세요.")
    else:
        y_test = st.session_state.splits["y_test"]

        available = list(st.session_state.proba.keys())
        if not available:
            st.info("학습된 모델이 없습니다. [4) 모델 학습]에서 학습을 먼저 실행하세요.")
        else:
            selected_models = st.multiselect("평가할 모델 선택", options=available, default=available)
            threshold = st.slider("분류 임계값(threshold)", 0.05, 0.95, 0.50, 0.01)

            rows = []
            roc_figs = []

            for m in selected_models:
                proba = st.session_state.proba[m]
                met = eval_binary_from_proba(y_test, proba, threshold=threshold)
                rows.append({
                    "Model": m,
                    "AUC": met["AUC"],
                    "Accuracy": met["Accuracy"],
                    "Precision": met["Precision"],
                    "Recall": met["Recall"],
                    "F1": met["F1"],
                    "TN": met["ConfusionMatrix"][0, 0],
                    "FP": met["ConfusionMatrix"][0, 1],
                    "FN": met["ConfusionMatrix"][1, 0],
                    "TP": met["ConfusionMatrix"][1, 1],
                })
                roc_figs.append((m, plot_roc(y_test, proba, title=f"ROC - {m} (AUC={met['AUC']:.3f})")))

            result_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
            st.write("모델 성능 요약")
            st.dataframe(result_df, use_container_width=True)

            st.write("ROC 곡선")
            for name, fig in roc_figs:
                st.pyplot(fig, clear_figure=True)

# -----------------------------
# Tab 6: Segmentation & default rate
# -----------------------------
with tabs[5]:
    st.subheader("6) PD 기반 고객세분화 / 부실율 / 전략 제시")

    if "y_test" not in st.session_state.splits:
        st.info("먼저 [2) 전처리 & 데이터 분할]을 완료하세요.")
    else:
        y_test = st.session_state.splits["y_test"]
        available = list(st.session_state.proba.keys())
        if not available:
            st.info("세분화를 위한 예측확률(PD)이 없습니다. [4) 모델 학습]을 먼저 실행하세요.")
        else:
            chosen = st.selectbox("세분화에 사용할 모델(PD)", options=available)
            n_bins = st.slider("등급 개수(분위수 기반)", 3, 10, 5, 1)

            proba = st.session_state.proba[chosen]
            seg_agg, seg_raw, labels = segment_table(y_test, proba, n_bins=int(n_bins))

            st.write("세그먼트 요약(Grade별 고객수/평균PD/관측부실율)")
            st.dataframe(seg_agg, use_container_width=True)

            fig1 = plot_default_rate_by_grade(seg_agg, title=f"Default Rate by Grade - {chosen}")
            st.pyplot(fig1, clear_figure=True)

            # PD distribution
            st.write("PD 분포(히스토그램)")
            fig2 = plt.figure()
            ax = fig2.add_subplot(111)
            ax.hist(seg_raw["PD"], bins=30)
            ax.set_xlabel("Predicted PD")
            ax.set_ylabel("Count")
            ax.set_title(f"PD Distribution - {chosen}")
            st.pyplot(fig2, clear_figure=True)

            # Strategy suggestion (template)
            st.markdown("### 세분화 기반 전략(예시 템플릿)")
            st.write(
                "아래 전략은 일반적인 신용 리스크 운영 예시입니다. "
                "보고서에서는 Grade별 부실율(관측) 및 평균 PD를 근거로 문장화하면 점수가 잘 나옵니다."
            )

            # Build a simple text mapping
            # Assume A is lowest risk
            grade_list = seg_agg["Grade"].tolist()
            if grade_list:
                low = grade_list[0]
                high = grade_list[-1]
                mid = grade_list[len(grade_list)//2]

                st.markdown(
                    f"""
- **{low} (저위험)**: 우대금리/한도 상향, 자동승인 비중 확대, 교차판매 타겟
- **{mid} (중위험)**: 기본정책 유지 + 조건부 승인(예: DTI, 소득 확인), 모니터링 강화
- **{high} (고위험)**: 심사 강화(추가서류/보증), 한도 축소, 금리 가산 또는 거절 기준 적용
"""
                )

            st.markdown("### 부실율(Observed Default Rate) 정의")
            st.code("부실율 = (해당 세그먼트의 실제 부실(1) 건수) / (해당 세그먼트 고객수)")

st.caption("주의: 본 코드는 과제/프로토타입용 기본 템플릿입니다. 실무 수준에서는 (1) 시점 정의, (2) 누수 변수 제거, (3) 훈련 통계(중앙값 등) 저장/재사용, (4) 교차검증, (5) 캘리브레이션, (6) 클래스 불균형 처리 등을 추가 권장합니다.")
