import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def ttest_filter_numeric(df: pd.DataFrame, target_col: str, p_threshold: float = 0.5):
    """수치형 변수만: y=0 vs y=1 t-test, p<=threshold 통과"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    tmp = df[[target_col] + num_cols].dropna(subset=[target_col])
    g0 = tmp[tmp[target_col] == 0]
    g1 = tmp[tmp[target_col] == 1]

    passed = []
    rows = []
    for c in num_cols:
        x0 = g0[c].dropna()
        x1 = g1[c].dropna()
        if len(x0) < 2 or len(x1) < 2:
            continue
        # Welch t-test
        stat, p = stats.ttest_ind(x0, x1, equal_var=False, nan_policy="omit")
        rows.append((c, float(p)))
        if p <= p_threshold:
            passed.append(c)

    res = pd.DataFrame(rows, columns=["feature", "p_value"]).sort_values("p_value")
    return passed, res

def remove_outliers_iqr(df: pd.DataFrame, cols: list[str], k: float = 1.5):
    """IQR 기반 이상치 행 제거(모든 cols에 대해 범위 밖이면 제거)"""
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    for c in cols:
        s = out[c]
        if not np.issubdtype(s.dtype, np.number):
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        mask &= s.between(lo, hi) | s.isna()
    return out.loc[mask].copy()

def preprocess_pipeline(df: pd.DataFrame, target_col: str, ttest_p: float = 0.5, iqr_k: float = 1.5):
    """(1) t-test 필터 → (2) 이상치 제거 → (3) 결측치 제거 → (4) 원핫 → (5) 스케일링"""
    # 타깃 결측 제거
    df0 = df.dropna(subset=[target_col]).copy()

    # t-test(수치형만) 통과 컬럼
    passed_num, ttest_table = ttest_filter_numeric(df0, target_col, p_threshold=ttest_p)

    # 전처리 대상 X 구성(수치형은 passed만 유지, 범주형은 전체 유지)
    num_cols = df0.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    cat_cols = [c for c in df0.columns if c not in ([target_col] + num_cols)]
    use_cols = passed_num + cat_cols

    X = df0[use_cols].copy()
    y = df0[target_col].astype(int).copy()

    # 이상치 제거(IQR) — t-test 통과 수치형에 대해서만
    if len(passed_num) > 0:
        tmp = pd.concat([X, y], axis=1)
        tmp2 = remove_outliers_iqr(tmp, passed_num, k=iqr_k)
        y = tmp2[target_col].astype(int)
        X = tmp2.drop(columns=[target_col])

    # 결측치 제거
    tmp3 = pd.concat([X, y], axis=1).dropna()
    y = tmp3[target_col].astype(int)
    X = tmp3.drop(columns=[target_col])

    # 원핫 인코딩
    X_oh = pd.get_dummies(X, drop_first=True)

    # 스케일링: (원래 수치형 passed_num에 해당하는 컬럼명만) 스케일
    scaler = StandardScaler()
    scale_cols = [c for c in X_oh.columns if c in passed_num]  # get_dummies는 수치형 컬럼명 유지
    if len(scale_cols) > 0:
        X_oh[scale_cols] = scaler.fit_transform(X_oh[scale_cols])

    return X_oh, y, ttest_table, passed_num, scaler

def forward_stepwise_logit(X: pd.DataFrame, y: pd.Series, p_enter: float = 0.05, max_steps: int = 200):
    """전진선택법(Forward): 가장 작은 p-value 변수를 단계적으로 추가"""
    remaining = list(X.columns)
    selected = []
    last_model = None

    for _ in range(min(max_steps, len(remaining))):
        best_p = None
        best_var = None
        best_model = None

        for v in remaining:
            cols_try = selected + [v]
            X_const = sm.add_constant(X[cols_try], has_constant="add")

            try:
                m = sm.Logit(y, X_const).fit(disp=False)
                p = float(m.pvalues.get(v, 1.0))
            except Exception:
                continue

            if (best_p is None) or (p < best_p):
                best_p = p
                best_var = v
                best_model = m

        if best_var is None or best_p is None or best_p > p_enter:
            break

        selected.append(best_var)
        remaining.remove(best_var)
        last_model = best_model

    return last_model, selected
