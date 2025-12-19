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

# ----------------------------
# Streamlit 기본 설정
# ----------------------------
st.set_page_config(page_title="신경망 기반 개인신용평가(부실예측)", layout="wide")
st.title("신경망(MLP) 기반 개인신용평가 모델")

# ----------------------------
# 유틸 함수
# ----------------------------
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
        "F1": f1_score(y_true, pred, zero_division=0),
        "CM": confusion_matrix(y_true, pred),
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

def make_quantile_grades(proba, n_bins=5):
    # 분위수 기반 위험등급 생성(낮음=A, 높음=...)
    s = pd.Series(proba)
    # 중복값이 많을 때 qcut 실패 방지: rank 사용
    r = s.rank(method="average")
    q = pd.qcut(r, q=n_bins, labels=False, duplicates="drop")
    actual_bins = int(pd.Series(q).nunique())
    labels = [chr(ord("A") + i) for i in range(actual_bins)]  # A,B,C...
    grade = pd.Series(q).map(lambda i: labels[int(i)] if pd.notna(i) else labels[-1])
    return grade, labels

def segmentation_table(y_true, proba, n_bins=5):
    grade, labels = make_quantile_grades(proba, n_bins=n_bins)
    temp = pd.DataFrame({"PD": proba, "Y": y_true, "Grade": grade})
    agg = temp.groupby("Grade").agg(
        Customers=("Y", "count"),
        Avg_PD=("PD", "mean"),
        Default_Rate=("Y", "mean")
    ).reset_index()
    agg["order"] = agg["Grade"].apply(lambda x: ord(x) - ord("A"))
    agg = agg.sort_values("order").drop(columns=["order"])
    return agg, temp

def plot_default_rate_by_grade(agg_df, title="Default Rate by Risk Grade"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(agg_df["Grade"], agg_df["Default_Rate"])
    ax.set_xlabel("Risk Grade (A=Low → High)")
    ax.set_ylabel("Observed Default Rate")
    ax.set_title(title)
    return fig


# ----------------------------
# 세션 상태
# ----------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "prep_pipe" not in st.session_state:
    st.session_state.prep_pipe = None
if "model" not in st.session_state:
    st.session_state.model = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "proba_test" not in st.session_state:
    st.session_state.proba_test = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None


# ----------------------------
# 데이터마이닝 절차 탭 구성
# ----------------------------
tabs = st.tabs([
    "1) 데이터 탐색(EDA)",
    "2) 데이터 전처리",
    "3) 모델링(신경망)",
    "4) 성능평가",
    "5) PD 기반 고객세분화/부실율"
])

# ============================================================
# 0) 데이터 업로드 (공통)
# ============================================================
st.sidebar.header("데이터 업로드")
uploaded = st.sidebar.file_uploader("CSV 업로드", type=["csv"])

if uploaded is not None:
    df = safe_read_csv(uploaded)
    st.session_state.df = df

df = st.session_state.df
if df is None:
    st.info("좌측 사이드바에서 CSV 파일을 업로드하세요.")
    st.stop()

# ============================================================
# 1) 데이터 이해(EDA)
# ============================================================
with tabs[0]:
    st.subheader("1) 데이터 탐색(EDA): 변수 확인, 기초통계, 타깃 분포")

    st.write("데이터 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    st.write("기초 통계(수치형)")
    st.dataframe(df.describe(include=[np.number]).T, use_container_width=True)

    # 타깃 변수: not.fully.paid 고정 + 디자인 유지(선택 UI는 유지하되 비활성화)
    if "not.fully.paid" not in df.columns:
        st.error("타깃 변수 'not.fully.paid' 컬럼이 데이터에 없습니다.")
        st.stop()

    default_target = "not.fully.paid"
    target_col = st.selectbox(
        "타깃(Y) 컬럼 선택",
        options=df.columns.tolist(),
        index=df.columns.tolist().index(default_target),
        disabled=True  # ✅ 선택 기능만 제거
    )
    st.session_state.target_col = target_col

    # 타깃 분포
    y_raw = df[target_col]
    st.write("타깃 분포")
    st.dataframe(
        y_raw.value_counts(dropna=False).rename_axis("value").to_frame("count"),
        use_container_width=True
    )

    st.caption("해석 포인트: 타깃이 이진(0/1)인지 확인하고, 결측치/이상치/범주형 변수를 파악합니다.")


# ============================================================
# 2) 데이터 전처리
# ============================================================
with tabs[1]:
    st.subheader("2) 데이터 전처리: 누수 방지, 결측치 처리, 인코딩, 스케일링, 분할")

    target_col = st.session_state.target_col
    if target_col is None:
        st.info("먼저 [1) 데이터 이해]에서 타깃을 선택하세요.")
        st.stop()

    # Feature 선택
    default_features = [c for c in df.columns if c != target_col]
    # 스크린샷 기반 추천(있으면 기본 선택)
    suggested = [
        "credit.policy","purpose","int.rate","installment","log.annual.inc","dti",
        "fico","days.with.cr.line","revol.bal","revol.util","inq.last.6mths",
        "delinq.2yrs","pub.rec"
    ]
    suggested = [c for c in suggested if c in df.columns]
    default_select = suggested if len(suggested) > 0 else default_features

    feature_cols = st.multiselect("설명변수(X) 선택", options=default_features, default=default_select)
    if len(feature_cols) == 0:
        st.warning("설명변수(X)를 최소 1개 이상 선택하세요.")
        st.stop()

    # 전처리 옵션
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test 비율", 0.1, 0.5, 0.2, 0.05)
    with col2:
        random_state = st.number_input("random_state", 0, 9999, 42, 1)
    with col3:
        stratify = st.checkbox("Stratify(Y) 적용", value=True)

    # 타깃 이진화 확인(필요 시)
    st.markdown("**타깃 이진화(필요한 경우만)**")
    bin_mode = st.radio("타깃 처리 방식", ["이미 0/1 이진", "특정 값들을 부실(1)로 지정"], index=0, horizontal=True)

    if bin_mode == "특정 값들을 부실(1)로 지정":
        pos_classes = st.multiselect("부실(1)로 볼 타깃 값 선택", options=sorted(df[target_col].astype(str).unique().tolist()))
    else:
        pos_classes = None

    # 전처리 파이프라인 구축(Train에서 fit)
    if st.button("전처리 + 분할 실행"):
        # y 만들기
        if bin_mode == "특정 값들을 부실(1)로 지정":
            if not pos_classes:
                st.error("부실(1) 값이 비어있습니다.")
                st.stop()
            y = df[target_col].astype(str).isin([str(x) for x in pos_classes]).astype(int).values
        else:
            # 0/1이라고 가정
            y = df[target_col].astype(int).values

        X = df[feature_cols].copy()

        # 수치/범주 컬럼 분리
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        st.write("수치형:", num_cols)
        st.write("범주형:", cat_cols)

        # 전처리: 결측치 처리 + (범주형 one-hot) + (수치형 스케일링)
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols)
            ],
            remainder="drop"
        )

        # split
        strat_y = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=int(random_state), stratify=strat_y
        )

        # fit/transform
        X_train_p = preprocessor.fit_transform(X_train)
        X_test_p = preprocessor.transform(X_test)

        st.session_state.prep_pipe = preprocessor
        st.session_state.X_train_p = X_train_p
        st.session_state.X_test = X_test_p
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_cols = feature_cols

        st.success("전처리 및 분할 완료")
        st.write("X_train shape:", X_train_p.shape, "X_test shape:", X_test_p.shape)
        st.write("y_train 분포:", pd.Series(y_train).value_counts().to_dict())
        st.write("y_test 분포:", pd.Series(y_test).value_counts().to_dict())

    if "X_train_p" in st.session_state:
        st.caption("전처리가 완료되었습니다. 다음 탭에서 신경망 모델을 학습하세요.")

# ============================================================
# 3) 모델링(신경망)
# ============================================================
with tabs[2]:
    st.subheader("3) 모델링(신경망): MLP 학습 및 예측확률(PD) 생성")

    if "X_train_p" not in st.session_state:
        st.info("먼저 [2) 데이터 전처리]에서 전처리+분할을 실행하세요.")
        st.stop()

    X_train_p = st.session_state.X_train_p
    y_train = st.session_state.y_train

    # 하이퍼파라미터
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        h1 = st.number_input("Hidden Layer 1", 16, 512, 64, 16)
    with c2:
        h2 = st.number_input("Hidden Layer 2 (0이면 1층)", 0, 512, 32, 16)
    with c3:
        alpha = st.number_input("L2 규제(alpha)", 0.0, 0.01, 0.0001, 0.0001, format="%.4f")
    with c4:
        max_iter = st.number_input("max_iter", 100, 5000, 500, 100)

    hidden = (int(h1),) if int(h2) == 0 else (int(h1), int(h2))

    colA, colB = st.columns(2)
    with colA:
        early_stopping = st.checkbox("early_stopping 사용", value=True)
    with colB:
        validation_fraction = st.slider("validation_fraction", 0.05, 0.30, 0.10, 0.01)

    if st.button("신경망 학습 실행"):
        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=float(alpha),
            max_iter=int(max_iter),
            random_state=42,
            early_stopping=bool(early_stopping),
            validation_fraction=float(validation_fraction) if early_stopping else 0.1
        )
        model.fit(X_train_p, y_train)

        st.session_state.model = model
        st.success("신경망 학습 완료")

        # test proba
        X_test_p = st.session_state.X_test
        proba_test = model.predict_proba(X_test_p)[:, 1]
        st.session_state.proba_test = proba_test

        st.write("예측확률(PD) 샘플(상위 10개)")
        st.write(pd.Series(proba_test).head(10))

        # 학습 수렴 정보
        if hasattr(model, "loss_curve_"):
            st.write("학습 loss_curve 길이:", len(model.loss_curve_))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(model.loss_curve_)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss Curve")
            st.pyplot(fig, clear_figure=True)

# ============================================================
# 4) 성능평가
# ============================================================
with tabs[3]:
    st.subheader("4) 성능평가: AUC, Accuracy, Precision/Recall/F1, 혼동행렬, ROC")

    if st.session_state.proba_test is None:
        st.info("먼저 [3) 모델링]에서 신경망을 학습하세요.")
        st.stop()

    y_test = st.session_state.y_test
    proba_test = st.session_state.proba_test

    threshold = st.slider("분류 임계값(threshold)", 0.05, 0.95, 0.50, 0.01)
    met = metrics_from_proba(y_test, proba_test, threshold=float(threshold))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC", f"{met['AUC']:.4f}")
    c2.metric("Accuracy", f"{met['Accuracy']:.4f}")
    c3.metric("Precision", f"{met['Precision']:.4f}")
    c4.metric("Recall", f"{met['Recall']:.4f}")
    c5.metric("F1", f"{met['F1']:.4f}")

    st.write("혼동행렬(Confusion Matrix) [ [TN FP], [FN TP] ]")
    st.write(met["CM"])

    fig = plot_roc(y_test, proba_test, title=f"ROC Curve (AUC={met['AUC']:.3f})")
    st.pyplot(fig, clear_figure=True)

    # 확률 분포
    st.write("예측확률(PD) 분포")
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.hist(proba_test, bins=30)
    ax2.set_xlabel("Predicted PD")
    ax2.set_ylabel("Count")
    ax2.set_title("PD Distribution (Test)")
    st.pyplot(fig2, clear_figure=True)

# ============================================================
# 5) PD 기반 고객세분화/부실율
# ============================================================
with tabs[4]:
    st.subheader("5) PD 기반 고객세분화/부실율(Observed Default Rate) + 전략 템플릿")

    if st.session_state.proba_test is None:
        st.info("먼저 [3) 모델링]에서 신경망을 학습하세요.")
        st.stop()

    y_test = st.session_state.y_test
    proba_test = st.session_state.proba_test

    n_bins = st.slider("위험등급 개수(분위수)", 3, 10, 5, 1)
    agg, raw = segmentation_table(y_test, proba_test, n_bins=int(n_bins))

    st.write("등급별 요약(고객수/평균PD/관측부실율)")
    st.dataframe(agg, use_container_width=True)

    fig = plot_default_rate_by_grade(agg, title="Observed Default Rate by Risk Grade")
    st.pyplot(fig, clear_figure=True)

    # 세분화 해석/전략(보고서 문장에 바로 사용 가능)
    st.markdown("### 전략 제안(보고서/발표용 템플릿)")
    grade_list = agg["Grade"].tolist()
    if grade_list:
        low = grade_list[0]
        high = grade_list[-1]
        mid = grade_list[len(grade_list)//2]

        st.write(
            f"""
- **{low}(저위험)**: 자동승인 확대, 우대금리/한도 상향, 교차판매 타겟
- **{mid}(중위험)**: 기본정책 + 조건부 승인(소득/DTI 확인), 모니터링 강화
- **{high}(고위험)**: 심사 강화(추가서류/보증), 한도 축소, 금리 가산 또는 거절 기준 적용
"""
        )

    st.markdown("### 부실율 정의")
    st.code("부실율(Observed Default Rate) = (해당 등급의 실제 부실(1) 건수) / (해당 등급 고객수)")

st.caption(
    "본 앱은 데이터마이닝 절차(이해→전처리→모델링→평가→세분화)를 신경망(MLP)로 구현한 과제/프로토타입 템플릿입니다. "
    "실제 리스크 모델링에서는 누수 변수 제거, 시점 정의, 캘리브레이션, 불균형 처리 등을 추가하는 것이 권장됩니다."
)
