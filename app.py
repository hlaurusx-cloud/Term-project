import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

import matplotlib.pyplot as plt


st.set_page_config(page_title="Neural Network (MLP) in Streamlit", layout="wide")
st.title("🧠 Neural Network (MLP) 二分类 - Streamlit 示例")

# -----------------------------
# 工具函数
# -----------------------------
def build_preprocess_pipeline(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor


def ensure_binary_y(y: pd.Series) -> pd.Series:
    # 若已经是0/1或布尔，直接处理
    if y.dropna().nunique() == 2:
        # 尝试把 bool / object 统一映射成 0/1
        uniques = list(y.dropna().unique())
        # 常见情况：['0','1'] 或 [0,1] 或 [False, True]
        # 统一：取排序后的第一个为0，第二个为1（如你有特定正类，可自行改）
        mapping = {uniques[0]: 0, uniques[1]: 1}
        return y.map(mapping).astype("Int64")
    else:
        raise ValueError("目标列Y不是二分类（唯一值数 != 2）。请确认Y列。")


def plot_confusion_matrix(cm, labels=("0", "1")):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    return fig


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return fig, roc_auc


# -----------------------------
# 侧边栏：数据加载
# -----------------------------
st.sidebar.header("1) 数据")
uploaded = st.sidebar.file_uploader("上传 CSV", type=["csv"])

if uploaded is None:
    st.info("请先在左侧上传 CSV 文件。")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("数据预览")
st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# 选择 Y / X
# -----------------------------
st.sidebar.header("2) 变量选择")
y_col = st.sidebar.selectbox("选择目标变量 Y（必须二分类）", options=df.columns)

x_candidates = [c for c in df.columns if c != y_col]
x_cols = st.sidebar.multiselect("选择特征变量 X", options=x_candidates, default=x_candidates)

if len(x_cols) == 0:
    st.warning("请至少选择一个特征变量 X。")
    st.stop()

# -----------------------------
# 划分数据
# -----------------------------
st.sidebar.header("3) 划分与参数")
test_size = st.sidebar.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("random_state", value=42, step=1)

# MLP 参数
hidden_layer_sizes = st.sidebar.text_input("隐藏层结构（用逗号）", value="64,32")
alpha = st.sidebar.number_input("L2 正则 alpha", value=0.0001, format="%.6f")
max_iter = st.sidebar.number_input("最大迭代 max_iter", value=300, step=50)
learning_rate_init = st.sidebar.number_input("学习率 learning_rate_init", value=0.001, format="%.6f")

try:
    hls = tuple(int(x.strip()) for x in hidden_layer_sizes.split(",") if x.strip())
    if len(hls) == 0:
        raise ValueError
except Exception:
    st.error("隐藏层结构输入不合法，例如：64,32 或 128,64,32")
    st.stop()

# -----------------------------
# 训练按钮
# -----------------------------
train_btn = st.button("🚀 训练神经网络（MLP）")

if not train_btn:
    st.stop()

# -----------------------------
# 训练流程
# -----------------------------
try:
    X = df[x_cols].copy()
    y_raw = df[y_col].copy()
    y = ensure_binary_y(y_raw)

    # 去掉 y 为空的行
    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    preprocessor = build_preprocess_pipeline(X_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=hls,
        alpha=float(alpha),
        max_iter=int(max_iter),
        learning_rate_init=float(learning_rate_init),
        random_state=int(random_state),
        early_stopping=True,
        n_iter_no_change=10
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("mlp", mlp)
    ])

    with st.spinner("训练中..."):
        model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    # MLPClassifier 支持 predict_proba
    y_prob = model.predict_proba(X_test)[:, 1]

    # 指标
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    roc_fig, roc_auc = plot_roc(y_test, y_prob)

    # -----------------------------
    # 展示结果
    # -----------------------------
    st.subheader("模型结果")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1-score", f"{f1:.4f}")
    c5.metric("ROC-AUC", f"{roc_auc:.4f}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.write("混淆矩阵（Confusion Matrix）")
        fig_cm = plot_confusion_matrix(cm, labels=("0", "1"))
        st.pyplot(fig_cm)

    with right:
        st.write("ROC 曲线")
        st.pyplot(roc_fig)

    st.markdown("---")
    st.subheader("预测明细（前 50 行）")
    out = X_test.copy()
    out["y_true"] = y_test.values
    out["y_pred"] = y_pred
    out["y_prob(1)"] = y_prob
    st.dataframe(out.head(50), use_container_width=True)

    st.success("完成。")

except Exception as e:
    st.error(f"训练或评估过程中发生错误：{e}")
