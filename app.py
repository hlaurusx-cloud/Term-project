# app.py
# 개인신용평가(상환예측) 로지스틱 + Stepwise(t-test 기반) + 고객 세분화 Streamlit 앱

import streamlit as st
import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, roc_auc_score
)

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# 유틸 함수: Stepwise Backward Elimination (t-test / p-value 기반)
# ------------------------------------------------------------
def stepwise_backward_logit(X, y, p_threshold=0.05, max_iter=30):
    """
    statsmodels.Logit + backward elimination
    - p-value가 큰 변수를 하나씩 제거
    - X, y는 내부에서 숫자형(float)으로 변환하고 NaN 처리
    """

    # 1) X, y를 숫자형으로 강제 변환
    #    (object, bool, string 다 숫자로 바꾸고 안 되면 NaN)
    X_num = X.copy()
    X_num = X_num.apply(pd.to_numeric, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")

    # 2) y가 NaN인 행 제거 (둘 다 같은 index만 사용)
    mask = ~y_num.isna()
    X_num = X_num.loc[mask]
    y_num = y_num.loc[mask]

    # 3) X의 NaN은 0으로 채우고, 둘 다 float로 캐스팅
    X_num = X_num.fillna(0).astype(float)
    y_num = y_num.astype(float)

    # 4) 상수항 추가 후 역시 float로
    X_const = sm.add_constant(X_num, has_constant="add")
    X_const = X_const.astype(float)

    cols = list(X_const.columns)
    removed = []

    # -----------------------------------------
    # Stepwise backward elimination 반복 시작
    # -----------------------------------------
    for _ in range(max_iter):
        # 여기서 y_num, X_const[cols]는 전부 float이어야 함
        model = sm.Logit(y_num, X_const[cols]).fit(disp=False)
        pvalues = model.pvalues

        # const 제외한 가장 큰 p-value 찾기
        pvalues_no_const = pvalues.drop("const", errors="ignore")
        worst_feature = pvalues_no_const.idxmax()
        worst_p = pvalues_no_const.max()

        # 제거 조건 체크
        if worst_p > p_threshold and len(cols) > 2:
            cols.remove(worst_feature)
            removed.append((worst_feature, worst_p))
        else:
            break

    # -----------------------------------------
    # 모든 제거 작업 끝 → 최종 모델 적합
    # -----------------------------------------
    final_model = sm.Logit(y_num, X_const[cols]).fit(disp=False)

    return final_model, cols, removed



# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="개인신용평가(Logit) – 상환예측",
    layout="wide"
)

st.title("📊 개인신용평가 – Logit (상환예측) + 고객세분화")

# ------------------------------------------------------------
# 1. 데이터 업로드
# ------------------------------------------------------------
st.sidebar.header("1. 데이터 업로드")
uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded is None:
    st.info("👈 왼쪽 사이드바에서 CSV 파일을 먼저 업로드하세요.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("### 📁 업로드된 데이터 미리보기")
st.dataframe(df.head())

# ------------------------------------------------------------
# 2. 타깃 변수/설정 선택
# ------------------------------------------------------------
st.sidebar.header("2. 변수 설정")

all_cols = df.columns.tolist()

target_col = st.sidebar.selectbox(
    "타깃 변수 (부실 여부 / 상환 상태)",
    options=all_cols,
    index=all_cols.index("loan_status") if "loan_status" in all_cols else 0
)

# 타깃이 문자형이면, 어느 값(라벨)을 '부실(1)'로 볼지 선택
if df[target_col].dtype == "object":
    st.sidebar.markdown("**타깃이 범주형입니다. 부실(=1)로 볼 값을 선택하세요.**")
    unique_vals = df[target_col].dropna().unique().tolist()
    positive_label = st.sidebar.selectbox(
        "부실(1)로 간주할 값(라벨)",
        options=unique_vals
    )
    y_raw = df[target_col].apply(lambda x: 1 if x == positive_label else 0)
else:
    # 이미 0/1 이라고 가정
    y_raw = df[target_col]
    positive_label = 1

st.sidebar.write("---")

test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.5, 0.3, 0.05)
p_threshold = st.sidebar.slider("Stepwise 제거 기준 p-value", 0.01, 0.2, 0.05, 0.01)
random_state = st.sidebar.number_input("Random State", 0, 9999, 42)

st.sidebar.write("---")
st.sidebar.header("3. 고객 세분화 설정")

segmentation_method = st.sidebar.radio(
    "세분화 방식 선택",
    ["수동 임계값(Threshold)", "분위수(Quantile) 기반"],
)

if segmentation_method == "수동 임계값(Threshold)":
    st.sidebar.markdown("예: 0.05, 0.15, 0.30, 0.50 등")
    th1 = st.sidebar.number_input("등급 A/B 경계 (예: 0.05)", 0.0, 1.0, 0.05, 0.01)
    th2 = st.sidebar.number_input("등급 B/C 경계 (예: 0.15)", 0.0, 1.0, 0.15, 0.01)
    th3 = st.sidebar.number_input("등급 C/D 경계 (예: 0.30)", 0.0, 1.0, 0.30, 0.01)
    th4 = st.sidebar.number_input("등급 D/E 경계 (예: 0.50)", 0.0, 1.0, 0.50, 0.01)
else:
    st.sidebar.markdown("분위수 기반 5개 그룹 (A~E)으로 자동 분할합니다.")

# ------------------------------------------------------------
# 3. 전처리: X, y 구성 및 더미변수 생성
# ------------------------------------------------------------
st.header("1️⃣ 전처리 및 변수 구성")

# 타깃 제외한 나머지 컬럼을 설명변수 후보로 사용
feature_cols = [c for c in all_cols if c != target_col]
X_raw = df[feature_cols].copy()
y = y_raw.copy()

st.markdown("#### 🔍 결측치 처리")
st.write("기본적으로 **결측치 행은 제거(dropna)** 합니다.")
data = pd.concat([X_raw, y], axis=1).dropna()
X_raw = data[feature_cols]
y = data[target_col] if target_col in data.columns else y.loc[data.index]

st.markdown("#### 🔢 범주형 변수 인코딩 (One-Hot)")
cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_raw.select_dtypes(exclude=["object", "category"]).columns.tolist()

st.write(f"- 수치형 변수 개수: {len(num_cols)}")
st.write(f"- 범주형 변수 개수: {len(cat_cols)}")

X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

st.write("인코딩 후 X의 shape:", X_encoded.shape)
st.dataframe(X_encoded.head())

# ------------------------------------------------------------
# 4. Train/Test Split (+ stratify 에러 대비)
# ------------------------------------------------------------
st.header("2️⃣ 학습/검증 데이터 분할")

# 타깃 분포 확인
st.markdown("#### 🔍 타깃(부실 여부) 분포")
class_counts = y.value_counts()
st.write(class_counts)

# 기본은 stratify=y 로 시도하되, 에러 나면 stratify=None 으로 fallback
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y   # 우선 계층 샘플링 시도
    )
except ValueError as e:
    st.warning(
        "⚠️ stratify=y 옵션으로 Train/Test를 나누는 과정에서 오류가 발생했습니다. "
        "타깃 클래스 중 일부가 너무 적을 수 있어요.\n"
        "→ stratify 없이(무작위 분할) 다시 시도합니다.\n\n"
        f"원본 오류 메시지(참고용): {e}"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=None
    )

st.write(f"- Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
st.write(f"- 변수 개수: {X_train.shape[1]}")


# ------------------------------------------------------------
# 5. Stepwise + Logit 모델 학습
# ------------------------------------------------------------
st.header("3️⃣ 로지스틱 회귀(Logit) + Stepwise(t-test 기반)")

with st.spinner("Stepwise backward elimination으로 변수 선택 중..."):
    model_final, selected_cols, removed_list = stepwise_backward_logit(
        X_train, y_train, p_threshold=p_threshold
    )

st.subheader("📌 최종 선택된 변수 목록")
st.write(selected_cols)

if removed_list:
    st.subheader("❌ 제거된 변수 (변수명, p-value)")
    removed_df = pd.DataFrame(removed_list, columns=["feature", "p_value"])
    st.dataframe(removed_df)
else:
    st.write("Stepwise 과정에서 제거된 변수가 없습니다.")

st.subheader("📄 최종 Logit 모델 요약 (statsmodels)")
st.text(model_final.summary().as_text())

# ------------------------------------------------------------
# 6. 예측 및 성능평가
# ------------------------------------------------------------
st.header("4️⃣ 모델 성능 평가")

# train/test 데이터에 같은 선택 변수만 사용
X_train_sel = sm.add_constant(X_train[selected_cols[1:]], has_constant="add")
X_test_sel = sm.add_constant(X_test[selected_cols[1:]], has_constant="add")

# 예측확률 (기본: 부실(=1)의 확률)
train_pred_prob = model_final.predict(X_train_sel)
test_pred_prob = model_final.predict(X_test_sel)

# 0.5 기준으로 이항 분류
test_pred_label = (test_pred_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, test_pred_label)
prec = precision_score(y_test, test_pred_label, zero_division=0)
rec = recall_score(y_test, test_pred_label, zero_division=0)
f1 = f1_score(y_test, test_pred_label, zero_division=0)
fpr, tpr, _ = roc_curve(y_test, test_pred_prob)
auc = roc_auc_score(y_test, test_pred_prob)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall", f"{rec:.3f}")
col4.metric("F1-score", f"{f1:.3f}")
col5.metric("ROC AUC", f"{auc:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, test_pred_label)
st.subheader("🔢 혼동행렬 (Confusion Matrix)")
cm_df = pd.DataFrame(
    cm,
    index=[f"실제 0(정상)", f"실제 1(부실)"],
    columns=[f"예측 0(정상)", f"예측 1(부실)"]
)
st.dataframe(cm_df)

# ROC Curve
st.subheader("📈 ROC Curve")
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC curve (AUC={auc:.3f})"))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
fig_roc.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    width=700,
    height=500
)
st.plotly_chart(fig_roc, use_container_width=True)

# Test set 예측확률 분포
st.subheader("📊 Test 데이터 예측확률 분포(부실 확률)")
hist_df = pd.DataFrame({
    "pred_prob": test_pred_prob,
    "actual": y_test.values
})
fig_hist = px.histogram(
    hist_df,
    x="pred_prob",
    color="actual",
    nbins=30,
    barmode="overlay",
    labels={"actual": "실제 부실 여부", "pred_prob": "부실 예측 확률"}
)
fig_hist.update_traces(opacity=0.6)
st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------------------
# 7. 전체 데이터에 대한 예측 + 고객 세분화
# ------------------------------------------------------------
st.header("5️⃣ 예측확률 기반 고객 세분화 및 부실율")

# 전체 데이터(결측 제거 후)에 대해 예측 수행
X_all_encoded = X_encoded.loc[X_train.index.union(X_test.index)]  # 이미 dropna 되었음
y_all = y.loc[X_all_encoded.index]

X_all_sel = X_all_encoded[selected_cols[1:]].copy()
X_all_sel = X_all_sel.apply(pd.to_numeric, errors="coerce").fillna(0)
X_all_sel = sm.add_constant(X_all_sel, has_constant="add")

all_pred_prob = model_final.predict(X_all_sel)

seg_df = pd.DataFrame({
    "pred_prob": all_pred_prob,
    "actual": y_all.values
})

# 세분화
if segmentation_method == "수동 임계값(Threshold)":
    def assign_segment(p):
        if p < th1:
            return "A (매우 우량)"
        elif p < th2:
            return "B (우량)"
        elif p < th3:
            return "C (주의)"
        elif p < th4:
            return "D (고위험)"
        else:
            return "E (매우 고위험)"
    seg_df["segment"] = seg_df["pred_prob"].apply(assign_segment)
else:
    # 분위수 기반 5개 그룹 (pred_prob 낮을수록 A, 높을수록 E)
    seg_df["segment"] = pd.qcut(
        seg_df["pred_prob"],
        5,
        labels=["A (매우 우량)", "B (우량)", "C (주의)", "D (고위험)", "E (매우 고위험)"]
    )

# 세그먼트별 부실율 계산
group_stats = seg_df.groupby("segment").agg(
    고객수=("actual", "count"),
    부실수=("actual", "sum"),
    부실율=("actual", "mean"),
    평균부실확률=("pred_prob", "mean")
).reset_index()

group_stats["부실율(%)"] = group_stats["부실율"] * 100
group_stats["평균부실확률(%)"] = group_stats["평균부실확률"] * 100

st.subheader("📋 세그먼트별 고객수, 부실율, 평균 예측확률")
st.dataframe(group_stats[["segment", "고객수", "부실수", "부실율(%)", "평균부실확률(%)"]])

# 부실율 바 차트
st.subheader("📊 세그먼트별 부실율 시각화")
fig_seg = px.bar(
    group_stats,
    x="segment",
    y="부실율(%)",
    text="부실율(%)",
    labels={"segment": "세그먼트", "부실율(%)": "부실율(%)"},
)
fig_seg.update_traces(texttemplate="%{text:.1f}", textposition="outside")
fig_seg.update_layout(yaxis=dict(range=[0, group_stats["부실율(%)"].max() * 1.2]))
st.plotly_chart(fig_seg, use_container_width=True)

# 세그먼트 비중 파이차트
st.subheader("🧩 세그먼트별 고객 비중")
fig_pie = px.pie(
    group_stats,
    names="segment",
    values="고객수",
    hole=0.3
)
st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------------------------
# 8. 전략 제안 텍스트
# ------------------------------------------------------------
st.header("6️⃣ 고객 세분화 기반 전략 제안 (요약 텍스트)")

st.markdown("""
- **A (매우 우량)**: 부실 확률이 매우 낮은 그룹 → **우대금리, 한도 확대, 리워드 제공** 가능  
- **B (우량)**: 안정적인 그룹 → **표준 금리 유지**, 장기 고객으로 육성  
- **C (주의)**: 평균 수준 이상의 리스크 → **모니터링 강화**, 소액/단기 위주 승인  
- **D (고위험)**: 높은 리스크 → **금리 인상, 보증/담보 요구**, 승인 기준 강화  
- **E (매우 고위험)**: 매우 높은 리스크 → **대출 거절 또는 매우 제한적인 승인** 권장  
""")
