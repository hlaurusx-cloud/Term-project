import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
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

def segmentation_table(y_true, proba, n_bins=10):
    import numpy as np
    import pandas as pd

    # 1️⃣ 强制 1D
    y_true = np.asarray(y_true).ravel()
    proba  = np.asarray(proba).ravel()

    # 2️⃣ 长度检查（关键）
    if len(y_true) != len(proba):
        raise ValueError(
            f"[segmentation_table] 长度不一致: y_true={len(y_true)}, proba={len(proba)}"
        )

    # 3️⃣ 分箱（按概率分位数）
    grade = pd.qcut(proba, q=n_bins, labels=False, duplicates="drop") + 1

    temp = pd.DataFrame({
        "PD": proba,
        "Y": y_true,
        "Grade": grade
    })

    agg = (
        temp.groupby("Grade")
        .agg(
            cnt=("Y", "size"),
            bad=("Y", "sum"),
            avg_pd=("PD", "mean")
        )
        .reset_index()
    )

    agg["bad_rate"] = agg["bad"] / agg["cnt"]

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
    st.session_state["X_test"] = None
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

    st.write("데이터 크기:", df.shape)
    st.write("데이터 미리보기")
    st.dataframe(df.head(5), use_container_width=True)
    
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

    
    # ------------------------------------------------------------
    # EDA 시각화 (교체 버전)
    # ------------------------------------------------------------
    st.markdown("## 📊 EDA 시각화")

    # 1️⃣ 타깃 변수 분포 (Count + 불균형 확인)
    st.markdown("### 1️⃣ 타깃 변수 분포")
    target_cnt = y_raw.value_counts().sort_index()
    target_ratio = (target_cnt / target_cnt.sum() * 100).round(2)

    fig, ax = plt.subplots()
    ax.bar(target_cnt.index.astype(str), target_cnt.values)
    ax.set_xlabel("Target (0 = 정상, 1 = 부실)")
    ax.set_ylabel("Count")
    ax.set_title("Target Distribution")
    st.pyplot(fig)

    st.dataframe(
        pd.DataFrame({"count": target_cnt, "ratio(%)": target_ratio}),
        use_container_width=True
    )

    st.caption(
        "해석: 1(부실)보다 0(정상)의 비율이 매우 큰 경우, "
        "로지스틱/신경망 등 분류 모델에서 예측 편향 및 성능지표 해석 오류가 발생할 수 있습니다."
    )

    # 2️⃣ 수치형 변수 선택 → 타깃별 분포 비교(Boxplot)
    st.markdown("### 2️⃣ 수치형 변수의 타깃별 분포 비교")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    if len(num_cols) == 0:
        st.warning("수치형 변수가 없습니다.")
    else:
        selected_var = st.selectbox("분포를 비교할 수치형 변수 선택", options=num_cols, key="eda_selected_num")

        tmp = df[[selected_var, target_col]].dropna()
        if tmp[target_col].nunique() == 2:
            g0 = tmp[tmp[target_col] == 0][selected_var]
            g1 = tmp[tmp[target_col] == 1][selected_var]

            fig, ax = plt.subplots()
            ax.boxplot([g0, g1], labels=["Target = 0", "Target = 1"])
            ax.set_title(f"{selected_var} : Target별 분포 비교")
            ax.set_ylabel(selected_var)
            st.pyplot(fig)

            st.caption(
                "해석: 두 그룹의 중앙값·분산 차이가 클수록 해당 변수는 부실 여부를 구분하는 데 유의미할 가능성이 있습니다."
            )

            # 3️⃣ 분포 진단 (왜도·첨도 + 정규성 참고)
            st.markdown("### 3️⃣ 분포 진단 (참고)")
            x = tmp[selected_var]
            st.write(f"- 왜도 (Skewness): {stats.skew(x):.4f}")
            st.write(f"- 첨도 (Kurtosis, fisher): {stats.kurtosis(x, fisher=True):.4f}")

            if len(x) >= 3:
                x_sample = x.sample(n=min(5000, len(x)), random_state=42)
                _, p_value = stats.shapiro(x_sample)
                st.write(f"- Shapiro-Wilk p-value (표본≤5000): {p_value:.6f}")

            st.caption(
                "참고: 정규성은 로지스틱 회귀의 필수 전제는 아니지만, 극단적 왜도/이상치는 계수 추정과 모델 안정성에 영향을 줄 수 있습니다."
            )
        else:
            st.info("타깃이 이진(0/1) 형태가 아니어서 타깃별 박스플롯 비교를 생략합니다.")

    # 4️⃣ 수치형 변수 상관관계 (다중공선성 확인)
    st.markdown("### 4️⃣ 수치형 변수 상관관계(Heatmap)")
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] >= 2:
        corr = num_df.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(corr.values)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        ax.set_title("Correlation Heatmap (Numeric Variables)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

        st.caption("해석: 상관계수가 매우 높은 변수 쌍은 다중공선성 문제를 유발할 수 있어, 변수 선택/축소가 필요할 수 있습니다.")
    else:
        st.info("상관관계를 계산할 수 있는 수치형 변수가 충분하지 않습니다.")


# ============================================================
# 2) 데이터 전처리 (Wizard-like / 단계 고정형)
# ① T-test (p<=0.05) -> 통과 feature만 표시 (수치형만)
# ② 전처리 버튼 -> 이상치(IQR,k=1.5)/결측치 제거 + 원핫 (스케일링은 ③에서)
# ③ 데이터 분할(8:2) + Train 기준 표준화
#    + "분모델 저장": Logit(수치형만) / MLP(원핫 포함 전체)
# ============================================================
with tabs[1]:
    st.subheader("2) 데이터 전처리")

    # -----------------------------
    # 상태 초기화 (Reset 버튼 없음)
    # -----------------------------
    if "done_1" not in st.session_state: st.session_state["done_1"] = False
    if "done_2" not in st.session_state: st.session_state["done_2"] = False
    if "done_3" not in st.session_state: st.session_state["done_3"] = False

    # -----------------------------
    # 데이터/타깃 확인 (과제 조건: not.fully.paid 고정)
    # -----------------------------
    if df is None:
        st.warning("먼저 데이터를 로드하세요.")
        st.stop()

    target_col = "not.fully.paid"
    st.session_state["target_col"] = target_col

    if target_col not in df.columns:
        st.error("타깃 변수 'not.fully.paid' 컬럼이 데이터에 없습니다.")
        st.stop()

    st.info(f"타깃(Y): {target_col}")

    # =========================================================
    # ① T-test (p<=0.05 고정)
    # =========================================================
    st.markdown("## ① T-test 기반 Feature 1차 선별")
    st.caption("수치형 변수만, not.fully.paid(0/1) 기준, p-value ≤ 0.05 통과")

    p_thr = 0.05
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_all = [c for c in num_cols_all if c != target_col]

    if not st.session_state["done_1"]:
        if st.button("T-test 실행 (p ≤ 0.05)"):
            g0 = df[df[target_col] == 0]
            g1 = df[df[target_col] == 1]

            rows = []
            passed = []

            for col in num_cols_all:
                x0 = g0[col].dropna()
                x1 = g1[col].dropna()
                if len(x0) < 2 or len(x1) < 2:
                    continue
                try:
                    _, p = stats.ttest_ind(x0, x1, equal_var=False, nan_policy="omit")
                except Exception:
                    continue

                rows.append((col, float(p)))
                if p <= p_thr:
                    passed.append(col)

            ttest_df = (
                pd.DataFrame(rows, columns=["feature", "p_value"])
                .sort_values("p_value")
                .reset_index(drop=True)
            )

            st.session_state["ttest_passed"] = passed
            st.session_state["ttest_table"] = ttest_df
            st.session_state["done_1"] = True
            st.rerun()

    # ✅ ① 결과는 항상 표시
    if st.session_state.get("done_1", False):
        passed = st.session_state.get("ttest_passed", [])
        st.success(f"✅ ① 완료: 통과 feature {len(passed)}개")
        st.markdown("### ✅ T-test 통과 feature 목록")
        st.write(passed if len(passed) > 0 else "통과 feature 없음")

        with st.expander("p-value 결과표 보기(선택)"):
            st.dataframe(
                st.session_state.get("ttest_table", pd.DataFrame()),
                use_container_width=True
            )

    st.divider()

    # =========================================================
    # ② 데이터 전처리 (버튼만 / IQR k=1.5 고정)
    # =========================================================
    st.markdown("## ② 데이터 전처리")
    st.caption("이상치 제거(IQR,k=1.5) + 결측치 제거 + 원핫 인코딩 (스케일링은 ③에서)")

    if not st.session_state.get("done_1", False):
        st.info("🔒 ① T-test를 완료하면 ②가 활성화됩니다.")
        st.stop()

    iqr_k = 1.5  # 고정 (설정 UI 없음)

    if not st.session_state.get("done_2", False):
        if st.button("데이터 전처리 실행"):
            passed_num = st.session_state.get("ttest_passed", [])
        
            # ✅ 핵심: purpose는 무조건 범주형으로 처리 (숫자코딩 되어 있어도 원핫되게)
            if "purpose" in df.columns:
                df["purpose"] = df["purpose"].astype(str)
        
            # -------------------------------------------------
            # X 구성: 수치형=passed_num + 범주형=전체(단, target 제외)
            #  - 범주형은 dtype 기반으로 잡는 게 가장 안정적
            # -------------------------------------------------
            numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            cat_cols = [c for c in cat_cols if c != target_col]
        
            # passed_num이 비어도 cat_cols로 진행 가능(단, cat_cols도 비면 stop)
            use_cols = list(dict.fromkeys(passed_num + cat_cols))  # 중복 제거+순서 유지
        
            if len(use_cols) == 0:
                st.error("전처리에 사용할 feature가 없습니다.")
                st.stop()
        
            X = df[use_cols].copy()
            y = df[target_col].astype(int).copy()
        
            # (1) IQR 이상치 제거: passed 수치형에만 적용
            if len(passed_num) > 0:
                tmp = pd.concat([X, y.rename(target_col)], axis=1)
                mask = pd.Series(True, index=tmp.index)
        
                for c in passed_num:
                    if c not in tmp.columns:
                        continue
                    s = tmp[c]
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    if pd.isna(iqr) or iqr == 0:
                        continue
                    lo = q1 - iqr_k * iqr
                    hi = q3 + iqr_k * iqr
                    mask &= s.between(lo, hi) | s.isna()
        
                tmp = tmp.loc[mask].copy()
                y = tmp[target_col].astype(int)
                X = tmp.drop(columns=[target_col])
        
            # (2) 결측치 제거(요청: 제거)
            tmp2 = pd.concat([X, y.rename(target_col)], axis=1).dropna()
            y = tmp2[target_col].astype(int)
            X = tmp2.drop(columns=[target_col])
        
            # (3) 원핫 인코딩(1회만)
            #     - 범주형(purpose 포함) → purpose_* 생성됨
            X_oh = pd.get_dummies(X, drop_first=True)
        
            # (4) 표준화 대상 수치형 컬럼 기록 (③에서 Train 기준 적용)
            #     - 원핫된 컬럼(purpose_*)은 자동으로 제외됨(0/1)
            scale_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
            st.session_state["X_processed"] = X_oh
            st.session_state["y_processed"] = y
            st.session_state["scale_cols"] = scale_cols
            st.session_state["scaler"] = None
        
            # (디버깅용: 필요하면 잠깐 켰다가 지우세요)
            # st.write("원핫 후 컬럼 예시:", [c for c in X_oh.columns if c.startswith("purpose_")][:10])
        
            st.session_state["done_2"] = True
            st.rerun()


    # ✅ ② 결과 항상 표시
    if st.session_state.get("done_2", False):
        Xp = st.session_state["X_processed"]
        yp = st.session_state["y_processed"]
        st.success("✅ ② 완료: 전처리 결과가 저장되어 있습니다.")
        st.write(f"전처리 후 X shape: {Xp.shape} / y length: {len(yp)}")

    st.divider()

    # =========================================================
    # ③ 데이터 분할(8:2) + 표준화(Train 기준) — MLP 전용
    # =========================================================
    st.markdown("## ③ 데이터 분할(8:2) + 표준화(Train 기준)")
    st.caption("Train/Test 분할 후, Train 기준으로 표준화하여 데이터 누수를 방지합니다. (MLP 저장)")
    
    if not st.session_state.get("done_2", False):
        st.info("🔒 ② 전처리를 완료하면 ③이 활성화됩니다.")
        st.stop()
    
    Xp = st.session_state["X_processed"]   # 전처리 후 전체 변수(원핫 포함)
    yp = st.session_state["y_processed"]
    
    test_size = 0.2
    st.write(f"분할 비율: Train {int((1-test_size)*100)}% / Test {int(test_size*100)}% (고정)")
    
    if not st.session_state.get("done_3", False):
        if st.button("데이터 분할 + 표준화(Train 기준) 저장"):
            # -----------------------------
            # A. 공통 분할 (항상 전체 변수 사용)
            # -----------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                Xp, yp, test_size=test_size, random_state=42, stratify=yp
            )
    
            # -----------------------------
            # B. 표준화(Train 기준) — 수치형만
            # -----------------------------
            scaler = StandardScaler()
    
            scale_cols = st.session_state.get("scale_cols", [])
            scale_cols = [c for c in scale_cols if c in X_train.columns]
    
            X_train_mlp = X_train.copy()
            X_test_mlp  = X_test.copy()
    
            if len(scale_cols) > 0:
                X_train_mlp[scale_cols] = scaler.fit_transform(X_train[scale_cols])
                X_test_mlp[scale_cols]  = scaler.transform(X_test[scale_cols])
    
            # -----------------------------
            # C. 저장 (MLP만)
            # -----------------------------
            st.session_state["X_train_mlp"] = X_train_mlp
            st.session_state["X_test_mlp"]  = X_test_mlp
            st.session_state["y_train"] = y_train
            st.session_state["y_test"]  = y_test
    
            st.session_state["cols_mlp"] = list(X_train_mlp.columns)
            st.session_state["scaler"] = scaler
            st.session_state["scale_cols_applied"] = scale_cols
    
            st.session_state["done_3"] = True
            st.rerun()
    
    # ✅ ③ 결과 표시 (MLP만)
    if st.session_state.get("done_3", False):
        required = ["X_train_mlp", "X_test_mlp", "cols_mlp"]
        missing = [k for k in required if k not in st.session_state or st.session_state.get(k) is None]
        if missing:
            st.warning("③ 결과가 불완전합니다. 버튼을 다시 눌러 저장하세요.")
            st.write("누락된 키:", missing)
            st.session_state["done_3"] = False
            st.stop()
    
        st.success("✅ ③ 완료: 분할 + 표준화(Train 기준) — MLP 저장 완료")
        st.write("MLP Train/Test:", st.session_state["X_train_mlp"].shape, "/", st.session_state["X_test_mlp"].shape)
    
        with st.expander("MLP 변수(항상 전체, purpose 원핫 포함) 보기"):
            st.write(st.session_state["cols_mlp"])

    




# ============================================================
# 3) 모델링(신경망): MLP
# ③ 단계(데이터 분할) 결과만 사용
# ============================================================
with tabs[2]:
    st.subheader("3) 모델링(신경망): MLP 학습 및 예측확률(PD) 생성")

    # --------------------------------------------------------
    # 가드: ③ 완료 여부
    # --------------------------------------------------------
    required = ["X_train_mlp", "X_test_mlp", "y_train", "y_test"]
    missing = [k for k in required if k not in st.session_state]
    
    if missing:
        st.info("먼저 [② 전처리 → ③ 데이터 분할]를 완료하세요. (MLP용 데이터가 아직 저장되지 않았습니다.)")
        st.stop()


    # --------------------------------------------------------
    # 세션에서 데이터 로드 (핵심)
    # --------------------------------------------------------
    X_train = st.session_state["X_train_mlp"]
    X_test  = st.session_state["X_test_mlp"]
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]

    # numpy 변환 (MLP 안정성)
    Xtr = X_train.to_numpy()
    Xte = X_test.to_numpy()

    st.write("Train shape:", Xtr.shape, " / Test shape:", Xte.shape)

    # --------------------------------------------------------
    # 하이퍼파라미터
    # --------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        h1 = st.number_input("Hidden Layer 1", 16, 512, 64, 16)
    with c2:
        h2 = st.number_input("Hidden Layer 2 (0이면 1층)", 0, 512, 32, 16)
    with c3:
        alpha = st.number_input("L2 규제(alpha)", 0.0, 0.01, 0.0001, 0.0001, format="%.4f")
    with c4:
        max_iter = st.number_input("max_iter", 200, 5000, 2000, 100)

    hidden = (int(h1),) if int(h2) == 0 else (int(h1), int(h2))

    early_stopping = st.checkbox("early_stopping 사용", value=True)
    validation_fraction = st.slider("validation_fraction", 0.05, 0.30, 0.10, 0.01)

    # --------------------------------------------------------
    # 학습
    # --------------------------------------------------------
    if st.button("MLP 학습 실행"):
        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=float(alpha),
            max_iter=int(max_iter),
            random_state=42,
            early_stopping=early_stopping,
            validation_fraction=float(validation_fraction) if early_stopping else 0.1
        )

        model.fit(Xtr, y_train)

        st.session_state["model"] = model
        st.success("MLP 학습 완료")

        # 예측 확률
        proba_test = model.predict_proba(Xte)[:, 1]
        st.session_state["proba_test"] = proba_test

        st.write("예측확률(PD) 샘플")
        st.write(pd.Series(proba_test).head(10))

        # loss curve
        if hasattr(model, "loss_curve_"):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(model.loss_curve_)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss Curve")
            st.pyplot(fig, clear_figure=True)

# ============================================================
# 4) 모델 평가 & Segmentation (PD 등급표)
# ============================================================
with tabs[3]:
    st.subheader("4) 모델 평가 및 PD Segmentation")

    # ------------------------------------------------------
    # Guard: 예측확률 존재 여부
    # ------------------------------------------------------
    required = ["y_test", "proba_test"]
    missing = [k for k in required if k not in st.session_state or st.session_state.get(k) is None]
    if missing:
        st.warning("먼저 MLP 모델을 학습하여 예측확률(PD)을 생성하세요.")
        st.write("누락된 키:", missing)
        st.stop()

    import numpy as np
    import pandas as pd

    y_test = np.asarray(st.session_state["y_test"]).ravel()
    proba_test = np.asarray(st.session_state["proba_test"]).ravel()

    # ------------------------------------------------------
    # Type/shape safety
    # ------------------------------------------------------
    if len(y_test) != len(proba_test):
        st.error(
            f"y_test({len(y_test)})와 proba_test({len(proba_test)}) 길이가 다릅니다.\n"
            "③(분할/표준화) 이후 MLP를 다시 학습하세요."
        )
        st.stop()

    # proba 범위 체크
    if np.any(np.isnan(proba_test)) or np.any(np.isinf(proba_test)):
        st.error("proba_test에 NaN 또는 Inf가 포함되어 있습니다. 모델을 다시 학습하세요.")
        st.stop()

    # 확률 클리핑(아주 드문 수치문제 방지)
    proba_test = np.clip(proba_test, 1e-12, 1 - 1e-12)

    # ------------------------------------------------------
    # 4-A) 성능 평가(지표)
    # ------------------------------------------------------
    st.markdown("## ✅ 4-A) 모델 성능 평가")

    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        roc_curve, precision_recall_curve
    )

    # Threshold 설정
    thr = st.slider("분류 임계값(Threshold)", 0.05, 0.95, 0.50, 0.01)
    y_pred = (proba_test >= thr).astype(int)

    # 주요 지표
    auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ROC-AUC", f"{auc:.4f}")
    c2.metric("PR-AUC(AP)", f"{pr_auc:.4f}")
    c3.metric("Accuracy", f"{acc:.4f}")
    c4.metric("Precision", f"{prec:.4f}")
    c5.metric("Recall", f"{rec:.4f}")
    c6.metric("F1", f"{f1:.4f}")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df, use_container_width=True)

    # Classification report
    with st.expander("Classification Report (상세)"):
        st.text(classification_report(y_test, y_pred, digits=4))

    # ROC Curve / PR Curve
    st.markdown("### ROC / PR Curve")
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    pr_p, pr_r, _ = precision_recall_curve(y_test, proba_test)

    import matplotlib.pyplot as plt

    colA, colB = st.columns(2)
    with colA:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(fpr, tpr)
        ax1.plot([0, 1], [0, 1], linestyle="--")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title(f"ROC Curve (AUC={auc:.4f})")
        st.pyplot(fig1, clear_figure=True)

    with colB:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(pr_r, pr_p)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"PR Curve (AP={pr_auc:.4f})")
        st.pyplot(fig2, clear_figure=True)

    # ------------------------------------------------------
    # (선택) KS 계산/표시 — 금융/신용평가에서 가산점
    # ------------------------------------------------------
    with st.expander("KS (선택)"):
        # KS = max(TPR - FPR)
        ks = float(np.max(tpr - fpr))
        st.write(f"KS Statistic: **{ks:.4f}**")

    st.divider()

    # ------------------------------------------------------
    # 4-B) PD Segmentation
    # ------------------------------------------------------
    st.markdown("## ✅ 4-B) PD Segmentation (Grade Table)")

    st.markdown("### 🔹 PD Segmentation 설정")
    n_bins = st.slider("등급 수 (Grade 개수)", 5, 20, 10, 1)

    # Segmentation 실행 (기존 함수 사용)
    try:
        agg, raw = segmentation_table(y_test=y_test, proba=proba_test, n_bins=int(n_bins))
    except TypeError:
        # 네 함수가 positional만 받는 경우 대비
        agg, raw = segmentation_table(y_test, proba_test, n_bins=int(n_bins))

    st.success("PD Segmentation Table 생성 완료")

    # 결과 표시
    st.markdown("### 📊 PD Segmentation Table")
    st.dataframe(agg, use_container_width=True)

    st.markdown("### 📄 개별 관측치 (샘플)")
    st.dataframe(raw.head(20), use_container_width=True)

   
# ============================================================
# 5) 고객 세분화 전략 제시 + 시각화 (PD 기반)
#   - 입력: y_test, proba_test (Tab3/4에서 생성된 것)
#   - 출력: Grade Table + Segment Table + 전략 + 시각화 3종
# ============================================================
with tabs[4]:
    st.subheader("5) 고객 세분화 전략 제시 + 시각화 (PD 기반)")

    # --------------------------------------------------------
    # Guard
    # --------------------------------------------------------
    required = ["y_test", "proba_test"]
    missing = [k for k in required if k not in st.session_state or st.session_state.get(k) is None]
    if missing:
        st.warning("먼저 MLP 모델을 학습하여 예측확률(PD)을 생성하세요.")
        st.write("누락된 키:", missing)
        st.stop()

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    y_test = np.asarray(st.session_state["y_test"]).ravel().astype(int)
    proba_test = np.asarray(st.session_state["proba_test"]).ravel().astype(float)

    if len(y_test) != len(proba_test):
        st.error(f"y_test({len(y_test)})와 proba_test({len(proba_test)}) 길이가 다릅니다.")
        st.stop()

    if np.any(np.isnan(proba_test)) or np.any(np.isinf(proba_test)):
        st.error("proba_test에 NaN/Inf가 포함되어 있습니다. 모델을 다시 학습하세요.")
        st.stop()

    proba_test = np.clip(proba_test, 1e-12, 1 - 1e-12)

    # --------------------------------------------------------
    # A) Grade 설정 + Risk Segment 구조(고정 + 개념도 표시)
    # --------------------------------------------------------
    st.markdown("### 5-A) PD 기반 고객 등급화(Grade) + Risk Segment 설정")

    n_bins = 14

    method = "분위수(qcut) 기반"
    
    # ✅ Risk Segment 비중 고정 (30/40/30)
    low_pct = 0.30
    mid_pct = 0.40
    high_pct = 0.30

    st.markdown(
        """
#### 📌 Risk Segment 구조(고정, 30/40/30)
"""
    )

    df_seg = pd.DataFrame({"y": y_test, "pd": proba_test})

    # Grade 생성 (낮은 PD → 낮은 Grade)
    try:
        if method.startswith("분위수"):
            df_seg["grade"] = pd.qcut(df_seg["pd"], q=int(n_bins), labels=False, duplicates="drop")
        else:
            df_seg["grade"] = pd.cut(df_seg["pd"], bins=int(n_bins), labels=False, include_lowest=True)
        df_seg["grade"] = df_seg["grade"].astype(int) + 1
    except Exception as e:
        st.error(f"Grade 생성 실패: {e}")
        st.stop()

    # --------------------------------------------------------
    # B) Grade Summary (보고서용 핵심 표)
    # --------------------------------------------------------
    st.markdown("### 5-B) Grade 요약 테이블")

    grade_summary = (
        df_seg.groupby("grade")
        .agg(
            n=("y", "size"),
            bad=("y", "sum"),
            bad_rate=("y", "mean"),
            avg_pd=("pd", "mean"),
            min_pd=("pd", "min"),
            max_pd=("pd", "max"),
        )
        .reset_index()
        .sort_values("grade")
    )
    grade_summary["share"] = grade_summary["n"] / grade_summary["n"].sum()
    grade_summary["cum_share"] = grade_summary["share"].cumsum()
    grade_summary["cum_bad"] = grade_summary["bad"].cumsum()
    grade_summary["cum_bad_rate"] = grade_summary["cum_bad"] / grade_summary["n"].cumsum()

    st.dataframe(grade_summary, use_container_width=True)

    # --------------------------------------------------------
    # C) Risk Segment (Low/Medium/High) - ✅ 슬라이더 제거, 고정 비중 사용
    # --------------------------------------------------------
    st.markdown("### 5-C) Risk Segment (Low / Medium / High) 결과")

    # 고객 누적 기준 컷 계산 (low_pct/high_pct 고정값 사용)
    tmp = grade_summary.copy()
    tmp["cum_n"] = tmp["n"].cumsum()
    total_n = tmp["n"].sum()

    low_cut_n = total_n * low_pct
    high_cut_n = total_n * (1 - high_pct)

    low_cut_grade = int(tmp.loc[tmp["cum_n"] >= low_cut_n, "grade"].iloc[0])
    high_cut_grade = int(tmp.loc[tmp["cum_n"] >= high_cut_n, "grade"].iloc[0])

    def assign_segment(g):
        if g <= low_cut_grade:
            return "Low Risk"
        elif g >= high_cut_grade:
            return "High Risk"
        else:
            return "Medium Risk"

    df_seg["segment"] = df_seg["grade"].apply(assign_segment)

    segment_summary = (
        df_seg.groupby("segment")
        .agg(
            n=("y", "size"),
            bad=("y", "sum"),
            bad_rate=("y", "mean"),
            avg_pd=("pd", "mean"),
        )
        .reset_index()
    )

    # 순서 정렬
    order = pd.Categorical(
        segment_summary["segment"],
        categories=["Low Risk", "Medium Risk", "High Risk"],
        ordered=True
    )
    segment_summary = segment_summary.assign(_ord=order).sort_values("_ord").drop(columns=["_ord"])
    segment_summary["share"] = segment_summary["n"] / segment_summary["n"].sum()

    # 컷 정보도 같이 보여주기(설명용)
    st.info(f"세그먼트 컷(Grade 기준): Low ≤ G{low_cut_grade} / High ≥ G{high_cut_grade} (비중 30/40/30 고정)")
    st.dataframe(segment_summary, use_container_width=True)

    # --------------------------------------------------------
    # D) 전략 제시 (표)
    # --------------------------------------------------------
    st.markdown("### 5-D) 고객 세분화 전략(예시)")

    strategy_df = pd.DataFrame([
        {
            "Segment": "Low Risk",
            "정의": "PD 낮음 / 부실률 낮음",
            "권장 전략": "우대금리, 한도 상향, 자동승인 비중 확대",
            "목표": "수익 극대화(우량 고객 유지/확대)"
        },
        {
            "Segment": "Medium Risk",
            "정의": "PD 중간 / 관리 필요",
            "권장 전략": "조건부 승인, 추가 심사, 모니터링 강화",
            "목표": "리스크 관리 + 선별적 수익"
        },
        {
            "Segment": "High Risk",
            "정의": "PD 높음 / 부실률 높음",
            "권장 전략": "대출 제한/거절, 담보·보증 요구, 금리 상향, 사후관리 강화",
            "목표": "손실 최소화(리스크 회피)"
        },
    ])
    st.dataframe(strategy_df, use_container_width=True)

    # --------------------------------------------------------
    # E) 시각화
    # --------------------------------------------------------
    st.markdown("### 5-E) 시각화")

    colA, colB = st.columns(2)

    # 1) Grade별 고객 수 분포
    with colA:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(grade_summary["grade"], grade_summary["n"])
        ax.set_xlabel("Grade (낮은 PD → 높은 PD)")
        ax.set_ylabel("고객 수")
        ax.set_title("Customer Count by Grade")
        st.pyplot(fig, clear_figure=True)

    # 2) Grade별 실제 부실률 vs 평균 PD
    with colB:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(grade_summary["grade"], grade_summary["bad_rate"], marker="o", label="Observed Bad Rate")
        ax.plot(grade_summary["grade"], grade_summary["avg_pd"], marker="o", label="Average PD")
        ax.set_xlabel("Grade")
        ax.set_ylabel("Rate")
        ax.set_title("Bad Rate vs Avg PD by Grade")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    # 3) 누적 고객비중 vs 누적 부실비중 (High PD부터)
    st.markdown("#### 누적 부실 포착 (Lift-like)")

    gs_desc = grade_summary.sort_values("grade", ascending=False).copy()
    gs_desc["share"] = gs_desc["n"] / gs_desc["n"].sum()
    gs_desc["bad_share"] = gs_desc["bad"] / max(gs_desc["bad"].sum(), 1)

    gs_desc["cum_share"] = gs_desc["share"].cumsum()
    gs_desc["cum_bad_share"] = gs_desc["bad_share"].cumsum()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gs_desc["cum_share"], gs_desc["cum_bad_share"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Cumulative Customer Share (High PD → Low PD)")
    ax.set_ylabel("Cumulative Bad Share")
    ax.set_title("Cumulative Bad Capture Curve")
    st.pyplot(fig, clear_figure=True)

    # --------------------------------------------------------
    # F) 다운로드
    # --------------------------------------------------------
    with st.expander("CSV 다운로드"):
        st.download_button(
            "Grade Summary CSV 다운로드",
            data=grade_summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="pd_grade_summary.csv",
            mime="text/csv"
        )
        st.download_button(
            "Segment Summary CSV 다운로드",
            data=segment_summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="pd_segment_summary.csv",
            mime="text/csv"
        )
