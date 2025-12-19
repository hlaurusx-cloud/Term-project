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
# ① T-test (p<=0.5) -> 통과 feature만 표시
# ② 전처리 버튼 -> 이상치/결측치 제거 + 원핫 + 스케일링
# ③ Feature Selection -> Stepwise(전진선택)만 + 8:2 분할
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
    # 타깃 확인
    # -----------------------------
    target_col = st.session_state.get("target_col", None)
    if target_col is None:
        st.warning("먼저 [EDA] 탭에서 타깃 변수를 설정해야 합니다.")
        st.stop()

    # 과제 조건: not.fully.paid 고정
    target_col = "not.fully.paid"
    if target_col not in df.columns:
        st.error("타깃 변수 'not.fully.paid' 컬럼이 데이터에 없습니다.")
        st.stop()

    st.info(f"타깃(Y): {target_col}")

    # =========================================================
    # ① T-test
    # =========================================================
    st.markdown("## ① T-test 기반 Feature 1차 선별")
    st.caption("수치형 변수만, not.fully.paid(0/1) 기준, p-value ≤ 0.5 통과")

    p_thr = 0.5
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols_all = [c for c in num_cols_all if c != target_col]

    if not st.session_state["done_1"]:
        if st.button("T-test 실행 (p ≤ 0.5)"):
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

            ttest_df = pd.DataFrame(rows, columns=["feature", "p_value"]).sort_values("p_value")

            st.session_state["ttest_passed"] = passed
            st.session_state["ttest_table"] = ttest_df
            st.session_state["done_1"] = True
            st.rerun()

    # ✅ ① 결과는 항상 표시(사라지지 않음)
    if st.session_state.get("done_1", False):
        passed = st.session_state.get("ttest_passed", [])
        st.success(f"✅ ① 완료: 통과 feature {len(passed)}개")
        st.markdown("### ✅ T-test 통과 feature 목록")
        st.write(passed if len(passed) > 0 else "통과 feature 없음")

        with st.expander("p-value 결과표 보기(선택)"):
            st.dataframe(st.session_state.get("ttest_table", pd.DataFrame()),
                         use_container_width=True)

    st.divider()

    # =========================================================
    # ② 데이터 전처리
    # =========================================================
    st.markdown("## ② 데이터 전처리")
    st.caption("이상치 제거(IQR) + 결측치 제거 + 원핫 인코딩 + 스케일링")

    if not st.session_state.get("done_1", False):
        st.info("🔒 ① T-test를 완료하면 ②가 활성화됩니다.")
        st.stop()

    iqr_k = st.slider("IQR 이상치 제거 강도(k)", 1.0, 3.0, 1.5, 0.1)

    if not st.session_state.get("done_2", False):
        if st.button("데이터 전처리 실행"):
            passed_num = st.session_state.get("ttest_passed", [])

            # X 구성: 수치형=passed_num + 범주형=전체(단, target 제외)
            numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in df.columns if (c not in numeric_all) and (c != target_col)]
            use_cols = passed_num + cat_cols

            if len(use_cols) == 0:
                st.error("전처리에 사용할 feature가 없습니다.")
                st.stop()

            X = df[use_cols].copy()
            y = df[target_col].astype(int).copy()

            # (1) 이상치 제거(IQR) - passed 수치형에만 적용
            if len(passed_num) > 0:
                tmp = pd.concat([X, y], axis=1)
                mask = pd.Series(True, index=tmp.index)

                for c in passed_num:
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
            tmp2 = pd.concat([X, y], axis=1).dropna()
            y = tmp2[target_col].astype(int)
            X = tmp2.drop(columns=[target_col])

            # (3) 원핫 인코딩
            X_oh = pd.get_dummies(X, drop_first=True)

            # (4) 스케일링: 수치형 passed 변수만
            scaler = StandardScaler()
            scale_cols = [c for c in X_oh.columns if c in passed_num]
            if len(scale_cols) > 0:
                X_oh[scale_cols] = scaler.fit_transform(X_oh[scale_cols])

            st.session_state["X_processed"] = X_oh
            st.session_state["y_processed"] = y
            st.session_state["scaler"] = scaler

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
    # ③ Feature Selection + 8:2 분할 (Stepwise Forward ONLY)
    # =========================================================
    st.markdown("## ③ Feature Selection + 데이터 분할(8:2)")
    st.caption("Train에서만 전진선택법을 수행하여 데이터 누수를 방지합니다.")

    if not st.session_state.get("done_2", False):
        st.info("🔒 ② 전처리를 완료하면 ③이 활성화됩니다.")
        st.stop()

    Xp = st.session_state["X_processed"]
    yp = st.session_state["y_processed"]

    # 먼저 8:2 분할(고정)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        Xp, yp, test_size=0.2, random_state=42, stratify=yp
    )
    st.write(f"분할 완료: Train {X_train_raw.shape} / Test {X_test_raw.shape}")

    p_enter = st.slider("Stepwise 진입 기준(p_enter)", 0.001, 0.50, 0.05, 0.001)

    if not st.session_state.get("done_3", False):
        if st.button("Stepwise 실행 + 8:2 저장"):
            remaining = list(X_train_raw.columns)
            selected = []
            final_model = None

            for _ in range(len(remaining)):
                best_p = None
                best_var = None
                best_model = None

                for v in remaining:
                    cols_try = selected + [v]
                    X_const = sm.add_constant(X_train_raw[cols_try], has_constant="add")
                    try:
                        m = sm.Logit(y_train, X_const).fit(disp=False)
                        pval = float(m.pvalues.get(v, 1.0))
                    except Exception:
                        continue

                    if best_p is None or pval < best_p:
                        best_p = pval
                        best_var = v
                        best_model = m

                if best_var is None or best_p is None or best_p > p_enter:
                    break
                    

                selected.append(best_var)
                remaining.remove(best_var)
                final_model = best_model

            if len(selected) == 0:
                st.warning("선택된 변수가 없습니다. p_enter를 완화하세요.")
                st.stop()

            # 선택된 변수로 train/test 구성
            X_train = X_train_raw[selected].copy()
            X_test = X_test_raw[selected].copy()

            # ✅ 다음 단계에서 AttributeError 방지: 반드시 key로 저장
            st.session_state["selected_cols"] = selected
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.session_state["logit_stepwise_model"] = final_model
            
            st.session_state.pop("proba_test", None)
            st.session_state.pop("model", None)

            st.session_state["done_3"] = True
            st.rerun()

    # ✅ ③ 결과 항상 표시
    if st.session_state.get("done_3", False):
        st.success("✅ ③ 완료: Stepwise + 8:2 분할 결과가 저장되어 있습니다.")
        st.write("선택 변수 수:", len(st.session_state["selected_cols"]))
        with st.expander("선택 변수 전체 보기"):
            st.write(st.session_state["selected_cols"])
        st.write("Train shape:", st.session_state["X_train"].shape, "/ Test shape:", st.session_state["X_test"].shape)

# ============================================================
# 3) 모델링(신경망): MLP
# Stepwise(③) 결과만 사용
# ============================================================
with tabs[2]:
    st.subheader("3) 모델링(신경망): MLP 학습 및 예측확률(PD) 생성")

    # --------------------------------------------------------
    # 가드: Stepwise 완료 여부
    # --------------------------------------------------------
    required = ["X_train", "X_test", "y_train", "y_test"]
    missing = [k for k in required if k not in st.session_state]

    if missing:
        st.info("먼저 [② 전처리 → ③ Stepwise]를 완료하세요.")
        st.stop()

    # --------------------------------------------------------
    # 세션에서 데이터 로드 (핵심)
    # --------------------------------------------------------
    X_train = st.session_state["X_train"]
    X_test  = st.session_state["X_test"]
    y_train = st.session_state["y_train"]
    y_test  = st.session_state["y_test"]

    # numpy 변환 (MLP 안정성)
    Xtr = X_train.values
    Xte = X_test.values

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

    # ======================================================
    # ✅ 세그멘테이션 호출 전 가드 (⭐你找的就是这个)
    # ======================================================
    required = ["y_test", "proba_test"]
    missing = [k for k in required if k not in st.session_state]

    if missing:
        st.warning("먼저 MLP 모델을 학습하여 예측확률(PD)을 생성하세요.")
        st.stop()

    y_test = st.session_state["y_test"]
    proba_test = st.session_state["proba_test"]

    # 타입 안전 가드
    import numpy as np
    y_test = np.asarray(y_test).ravel()
    proba_test = np.asarray(proba_test).ravel()

    if len(y_test) != len(proba_test):
        st.error(
            f"y_test({len(y_test)})와 proba_test({len(proba_test)}) 길이가 다릅니다.\n"
            "③ Stepwise 이후 MLP를 다시 학습하세요."
        )
        st.stop()


    # ======================================================
    # Segmentation 설정
    # ======================================================
    st.markdown("### 🔹 PD Segmentation 설정")
    n_bins = st.slider("등급 수 (Grade 개수)", 5, 20, 10, 1)

    # ======================================================
    # Segmentation 실행
    # ======================================================
    agg, raw = segmentation_table(
        y_test,
        proba_test,
        n_bins=int(n_bins)
    )

    st.success("PD Segmentation Table 생성 완료")

    # ======================================================
    # 결과 표시
    # ======================================================
    st.markdown("### 📊 PD Segmentation Table")
    st.dataframe(agg, use_container_width=True)

    st.markdown("### 📄 개별 관측치 (샘플)")
    st.dataframe(raw.head(20), use_container_width=True)
