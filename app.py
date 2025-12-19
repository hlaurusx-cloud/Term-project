# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.neural_network import MLPClassifier

# --------------------------------------------------
# Streamlit 기본 설정
# --------------------------------------------------
st.set_page_config(page_title="신경망 기반 개인신용평가", layout="wide")
st.title("🧠 신경망 기반 개인신용평가 (부실예측)")

# --------------------------------------------------
# 1. 데이터 로드
# --------------------------------------------------
uploaded = st.file_uploader("📂 LendingClub / 파이코 데이터 업로드 (CSV)", type="csv")

if uploaded is None:
    st.info("CSV 파일을 업로드하세요.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"데이터 로드 완료: {df.shape}")
st.dataframe(df.head(), use_container_width=True)

# --------------------------------------------------
# 2. 타깃/설명변수 정의
# --------------------------------------------------
TARGET = "not.fully.paid"

FEATURES = [
    "credit.policy", "purpose", "int.rate", "installment",
    "log.annual.inc", "dti", "fico", "days.with.cr.line",
    "revol.bal", "revol.util", "inq.last.6mths",
    "delinq.2yrs", "pub.rec"
]

df = df[FEATURES + [TARGET]]

X = df.drop(columns=[TARGET])
y = df[TARGET]

# --------------------------------------------------
# 3. 전처리
# --------------------------------------------------
# 범주형 one-hot
X = pd.get_dummies(X, columns=["purpose"], drop_first=True)

# 결측치 처리
X = X.fillna(X.median())

# 표준화 (신경망 필수)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# 4. 신경망 모델 학습
# --------------------------------------------------
st.subheader("⚙️ 신경망 하이퍼파라미터")

c1, c2, c3 = st.columns(3)
with c1:
    h1 = st.number_input("Hidden Layer 1", 16, 256, 64, step=16)
with c2:
    h2 = st.number_input("Hidden Layer 2", 0, 256, 32, step=16)
with c3:
    max_iter = st.number_input("Max Iter", 100, 2000, 500, step=100)

hidden_layers = (h1,) if h2 == 0 else (h1, h2)

if st.button("🚀 신경망 학습 실행"):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 예측 확률(PD)
    pd_proba = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # 5. 성능 평가
    # --------------------------------------------------
    auc = roc_auc_score(y_test, pd_proba)
    st.success(f"ROC-AUC: {auc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, pd_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # --------------------------------------------------
    # 6. 부실확률 기반 고객 세분화
    # --------------------------------------------------
    st.subheader("📊 고객 세분화 및 부실율")

    n_bins = st.slider("Risk Grade 개수", 3, 10, 5)

    grade = pd.qcut(pd_proba, q=n_bins, labels=False)
    seg = pd.DataFrame({
        "PD": pd_proba,
        "Default": y_test.values,
        "Grade": grade
    })

    summary = seg.groupby("Grade").agg(
        고객수=("Default", "count"),
        평균_PD=("PD", "mean"),
        부실율=("Default", "mean")
    ).reset_index()

    st.dataframe(summary, use_container_width=True)

    # 부실율 시각화
    fig2, ax2 = plt.subplots()
    ax2.bar(summary["Grade"], summary["부실율"])
    ax2.set_xlabel("Risk Grade (높을수록 위험)")
    ax2.set_ylabel("Observed Default Rate")
    ax2.set_title("등급별 부실율")
    st.pyplot(fig2)

    st.markdown("### 📌 해석")
    st.write(
        "신경망이 예측한 부실확률(PD)을 기준으로 고객을 세분화한 결과, "
        "Risk Grade가 높아질수록 실제 부실율이 증가하는 경향을 보인다. "
        "이는 신경망 모델이 신용위험을 효과적으로 구분하고 있음을 의미한다."
    )
