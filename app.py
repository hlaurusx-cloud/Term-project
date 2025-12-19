# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# --------------------------------------------------
# Streamlit 기본 설정
# --------------------------------------------------
st.set_page_config(page_title="신경망 기반 개인신용평가(부실예측)", layout="wide")
st.title("신경망(MLP) 기반 개인신용평가 모델 – 데이터 마이닝 절차")

# --------------------------------------------------
# 세션 상태 초기화
# --------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0

for k in [
    "df", "target_col", "feature_cols", "preprocessor",
    "X_train_p", "X_test_p", "y_train", "y_test",
    "model", "proba_test"
]:
    if k not in st.session_state:
        st.session_state[k] = None

# --------------------------------------------------
# 유틸 함수
# --------------------------------------------------
def safe_read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw), encoding_errors="ignore")


def metrics_from_proba(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, proba),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred),
        "CM": confusion_matrix(y_true, pred)
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


def segmentation_table(y_true, proba, n_bins=5):
    s = pd.Series(proba).rank(method="average")
    q = pd.qcut(s, q=n_bins, labels=False, duplicates="drop")
    labels = [chr(ord("A") + i) for i in range(q.nunique())]

    grade = q.map(lambda x: labels[int(x)])
    df = pd.DataFrame({"PD": proba, "부실여부": y_true, "등급": grade})

    agg = df.groupby("등급").agg(
        고객수=("부실여부", "count"),
        평균_PD=("PD", "mean"),
        부실율=("부실여부", "mean")
    ).reset_index()

    agg["순서"] = agg["등급"].apply(lambda x: ord(x) - ord("A"))
    agg = agg.sort_values("순서").drop(columns="순서")
    return agg


def plot_default_rate_by_grade(agg_df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(agg_df["등급"], agg_df["부실율"])
    ax.set_xlabel("위험 등급 (A = 저위험 → 고위험)")
    ax.set_ylabel("관측 부실율")
    ax.set_title("위험 등급별 부실율")
    return fig


# --------------------------------------------------
# Sidebar: 데이터 업로드
# --------------------------------------------------
st.sidebar.header("데이터 업로드")
uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded is not None:
    st.session_state.df = safe_read_csv(uploaded)

df = st.session_state.df
if df is None:
    st.info("좌측 사이드바에서 CSV 파일을 업로드하세요.")
    st.stop()

# --------------------------------------------------
# 상단 단계 네비게이션 (동일 간격)
# --------------------------------------------------
steps = [
    "데이터 이해(EDA)",
    "데이터 전처리",
    "모델링(신경망)",
    "성능 평가",
    "PD 기반 고객세분화/부실율"
]

cols = st.columns(len(steps), gap="large")
for i, col in enumerate(cols):
    with col:
        if i == st.session_state.step:
            st.markdown(
                f"""
                <div style="
                    background:#ff4b4b;
                    color:white;
                    padding:10px;
                    text-align:center;
                    border-radius:8px;
                    font-weight:bold;">
                    {steps[i]}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            if st.button(steps[i], key=f"step_{i}", use_container_width=True):
                st.session_state.step = i
                st.rerun()

st.markdown("<hr/>", unsafe_allow_html=True)

# ==================================================
# 1) 데이터 이해(EDA)
# ==================================================
if st.session_state.step == 0:
    st.subheader("1) 데이터 이해(EDA)")

    st.write("데이터 미리보기 (상위 20행)")
    st.dataframe(df.head(20), use_container_width=True)

    st.write("컬럼 리스트")
    st.code(", ".join(df.columns.tolist()))

    st.write("수치형 변수 기초 통계")
    st.dataframe(df.describe(include=[np.number]).T, use_container_width=True)

    default_target = "not.fully.paid" if "not.fully.paid" in df.columns else df.columns[-1]
    target_col = st.selectbox(
        "타깃 변수(Y) 선택",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(default_target)
    )
    st.session_state.target_col = target_col

    st.write("타깃 변수 분포")
    st.dataframe(
        df[target_col].value_counts().rename_axis("값").to_frame("빈도"),
        use_container_width=True
    )

    st.write("결측치 개수 (상위 30개)")
    miss = df.isna().sum().sort_values(ascending=False).head(30)
    st.dataframe(miss.rename("결측치 수").to_frame(), use_container_width=True)

# ==================================================
# 2) 데이터 전처리
# ==================================================
elif st.session_state.step == 1:
    st.subheader("2) 데이터 전처리")

    target_col = st.session_state.target_col
    if target_col is None:
        st.warning("먼저 [데이터 이해] 단계에서 타깃 변수를 선택하세요.")
        st.stop()

    feature_cols = st.multiselect(
        "설명 변수(X) 선택",
        options=[c for c in df.columns if c != target_col]
    )
    if not feature_cols:
        st.warning("설명 변수를 최소 1개 이상 선택하세요.")
        st.stop()

    test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2)
    stratify = st.checkbox("타깃 기준 층화 추출", value=True)

    if st.button("전처리 및 데이터 분할 실행"):
        X = df[feature_cols]
        y = df[target_col].astype(int)

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ])

        strat = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=strat
        )

        st.session_state.X_train_p = preprocessor.fit_transform(X_train)
        st.session_state.X_test_p = preprocessor.transform(X_test)
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.preprocessor = preprocessor

        st.success("전처리 및 데이터 분할 완료")

# ==================================================
# 3) 모델링(신경망)
# ==================================================
elif st.session_state.step == 2:
    st.subheader("3) 모델링 – 신경망(MLP)")

    if st.session_state.X_train_p is None:
        st.warning("먼저 데이터 전처리를 수행하세요.")
        st.stop()

    h1 = st.number_input("은닉층 1 노드 수", 16, 512, 64, 16)
    h2 = st.number_input("은닉층 2 노드 수 (0이면 1층)", 0, 512, 32, 16)

    if st.button("신경망 모델 학습"):
        hidden = (h1,) if h2 == 0 else (h1, h2)
        model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, random_state=42)
        model.fit(st.session_state.X_train_p, st.session_state.y_train)

        st.session_state.model = model
        st.session_state.proba_test = model.predict_proba(st.session_state.X_test_p)[:, 1]

        st.success("신경망 모델 학습 완료")

# ==================================================
# 4) 성능 평가
# ==================================================
elif st.session_state.step == 3:
    st.subheader("4) 성능 평가")

    if st.session_state.proba_test is None:
        st.warning("먼저 모델을 학습하세요.")
        st.stop()

    met = metrics_from_proba(
        st.session_state.y_test,
        st.session_state.proba_test
    )

    st.metric("ROC-AUC", f"{met['AUC']:.4f}")
    st.write("혼동행렬")
    st.write(met["CM"])

    fig = plot_roc(st.session_state.y_test, st.session_state.proba_test)
    st.pyplot(fig)

# ==================================================
# 5) PD 기반 고객세분화 / 부실율
# ==================================================
elif st.session_state.step == 4:
    st.subheader("5) PD 기반 고객세분화 및 부실율")

    if st.session_state.proba_test is None:
        st.warning("먼저 모델을 학습하세요.")
        st.stop()

    n_bins = st.slider("위험 등급 개수", 3, 10, 5)
    agg = segmentation_table(
        st.session_state.y_test,
        st.session_state.proba_test,
        n_bins=n_bins
    )

    st.dataframe(agg, use_container_width=True)

    fig = plot_default_rate_by_grade(agg)
    st.pyplot(fig)

    st.code("부실율 = (해당 등급의 실제 부실 건수) / (해당 등급 고객 수)")
