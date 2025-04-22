#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XGBoost 详细使用指南与示例

XGBoost（eXtreme Gradient Boosting）是一种高效、灵活且可移植的分布式梯度提升库。
它在监督学习算法的框架下实现了机器学习算法，被设计成高效、灵活和便携的。
XGBoost提供了并行树提升（也称为GBDT、GBM），解决了许多数据科学问题。

主要特点:
1. 速度快：比传统GBM快约10倍
2. 并行计算：支持多核并行
3. 正则化：内置正则化，防止过拟合
4. 高准确性：在各种竞赛中表现优异
5. 灵活性：支持自定义目标函数和评估指标
6. 缺失值处理：能自动处理缺失值
7. 树剪枝：自底向上剪枝，避免过拟合
8. 内置交叉验证：方便超参数调优

本示例将全面介绍XGBoost的使用方法，从基础到高级。

作者：AI助手
日期：2023年
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.datasets import load_boston, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可重复
np.random.seed(42)

print("=" * 80)
print("XGBoost 详细使用指南与示例")
print("=" * 80)

# =============================================================================
# 第1部分：XGBoost基础 - 回归问题
# =============================================================================
print("\n第1部分：XGBoost基础 - 回归问题")
print("-" * 50)

"""
XGBoost可以通过两种方式使用：
1. Scikit-Learn API: 更符合Python用户的使用习惯，与sklearn兼容
2. 原生XGBoost API: 更灵活，性能更高，支持更多高级功能

我们先从Scikit-Learn API开始介绍，因为它更易于理解。
"""

# 步骤1：准备数据
# -----------------------------------------------------------------------------
# 加载波士顿房价数据集
try:
    # sklearn 1.0+版本已弃用load_boston
    boston = load_boston()
except:
    # 如果load_boston不可用，使用模拟数据
    print("注意: 使用模拟的回归数据集(sklearn已弃用波士顿房价数据集)")
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=506, n_features=13, noise=10, random_state=42)
    boston = {'data': X, 'target': y}

X, y = boston['data'], boston['target']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 步骤2：使用Scikit-Learn API的XGBoost
# -----------------------------------------------------------------------------
print("\n使用Scikit-Learn API的XGBoost回归")

# 初始化XGBoost回归模型
# 1. n_estimators: 树的数量（提升轮数）
# 2. max_depth: 树的最大深度，控制复杂度
# 3. learning_rate: 学习率(eta)，控制每棵树的权重
# 4. subsample: 用于训练每棵树的样本比例，用于防止过拟合
# 5. colsample_bytree: 用于训练每棵树的特征比例
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# 训练模型
xgb_model.fit(X_train, y_train)

# 预测
y_pred = xgb_model.predict(X_test)

# 评估模型 - 使用RMSE评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# 步骤3：使用原生XGBoost API
# -----------------------------------------------------------------------------
print("\n使用原生XGBoost API的回归")

# 转换数据为DMatrix格式
# DMatrix是XGBoost专用的数据格式，更高效
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'reg:squarederror',  # 目标函数：回归问题的平方误差
    'max_depth': 3,                   # 树的最大深度
    'eta': 0.1,                       # 学习率
    'subsample': 0.8,                 # 样本采样比例
    'colsample_bytree': 0.8,          # 特征采样比例
    'seed': 42                        # 随机种子
}

# 训练模型
# 1. params: 参数字典
# 2. dtrain: 训练数据
# 3. num_boost_round: 提升轮数（树的数量）
# 4. evals: 评估数据列表，格式为[(dtrain, 'train'), (dtest, 'test')]
# 5. early_stopping_rounds: 如果评估指标在多少轮后没有改善，则停止训练
# 6. verbose_eval: 是否打印评估信息
evals_result = {}  # 存储评估结果
num_round = 100
watchlist = [(dtrain, 'train'), (dtest, 'test')]
native_model = xgb.train(
    params,
    dtrain,
    num_round,
    evals=watchlist,
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=False
)

# 预测
native_pred = native_model.predict(dtest)

# 评估模型
native_rmse = np.sqrt(mean_squared_error(y_test, native_pred))
print(f"RMSE (原生API): {native_rmse:.4f}")

# 可视化训练过程
plt.figure(figsize=(10, 6))
plt.plot(evals_result['train']['rmse'], label='训练集')
plt.plot(evals_result['test']['rmse'], label='测试集')
plt.xlabel('迭代次数')
plt.ylabel('RMSE')
plt.title('训练过程中的RMSE变化')
plt.legend()
plt.grid()
plt.savefig('xgb_training_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("训练曲线已保存为 'xgb_training_curve.png'")

# =============================================================================
# 第2部分：XGBoost基础 - 分类问题
# =============================================================================
print("\n第2部分：XGBoost基础 - 分类问题")
print("-" * 50)

# 步骤1：准备数据
# -----------------------------------------------------------------------------
# 加载乳腺癌数据集 (二分类问题)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 步骤2：使用Scikit-Learn API的XGBoost进行分类
# -----------------------------------------------------------------------------
print("\n使用Scikit-Learn API的XGBoost分类")

# 初始化XGBoost分类模型
# 对于二分类问题：
# 1. objective='binary:logistic': 输出概率
# 2. eval_metric='logloss': 使用对数损失评估
# 3. use_label_encoder=False: 避免警告(>=1.6.0版本)
try:
    # 较新版本XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
except:
    # 兼容旧版本XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

# 训练模型
xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

# 预测
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]  # 获取正类的概率
y_pred = xgb_clf.predict(X_test)

# 评估模型
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"准确率: {acc:.4f}")
print(f"AUC: {auc:.4f}")

# 步骤3：使用原生XGBoost API进行分类
# -----------------------------------------------------------------------------
print("\n使用原生XGBoost API的分类")

# 转换数据为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'binary:logistic',  # 目标函数：二分类的逻辑回归
    'max_depth': 3,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',        # 评估指标：对数损失
    'seed': 42
}

# 训练模型
evals_result = {}
num_round = 100
watchlist = [(dtrain, 'train'), (dtest, 'test')]
native_clf = xgb.train(
    params,
    dtrain,
    num_round,
    evals=watchlist,
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=False
)

# 预测
native_pred_proba = native_clf.predict(dtest)
native_pred = (native_pred_proba > 0.5).astype(int)

# 评估模型
native_acc = accuracy_score(y_test, native_pred)
native_auc = roc_auc_score(y_test, native_pred_proba)
print(f"准确率 (原生API): {native_acc:.4f}")
print(f"AUC (原生API): {native_auc:.4f}")

# =============================================================================
# 第3部分：多分类问题
# =============================================================================
print("\n第3部分：XGBoost - 多分类问题")
print("-" * 50)

# 创建一个多分类数据集
X_multi, y_multi = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15, 
    n_redundant=5, 
    n_classes=3, 
    random_state=42
)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# 使用XGBoost多分类
print("\n使用XGBoost进行多分类")

# 初始化XGBoost多分类模型
# 对于多分类问题：
# 1. objective='multi:softprob': 输出多类概率
# 2. num_class=3: 类别数量
xgb_multi = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    objective='multi:softprob',
    num_class=3,
    random_state=42
)

# 训练模型
xgb_multi.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

# 预测
y_pred = xgb_multi.predict(X_test)
y_pred_proba = xgb_multi.predict_proba(X_test)

# 评估模型
acc_multi = accuracy_score(y_test, y_pred)
print(f"多分类准确率: {acc_multi:.4f}")

# =============================================================================
# 第4部分：特征重要性
# =============================================================================
print("\n第4部分：XGBoost - 特征重要性")
print("-" * 50)

# 获取特征重要性 (Scikit-Learn API)
importances = xgb_clf.feature_importances_
feature_names = cancer.feature_names

# 创建DataFrame，方便可视化
feat_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})
feat_importance = feat_importance.sort_values('importance', ascending=False)

# 打印前10个重要特征
print("\n前10个重要特征:")
print(feat_importance.head(10))

# 可视化特征重要性
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feat_importance.head(10))
plt.title('XGBoost 特征重要性')
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("特征重要性图已保存为 'xgb_feature_importance.png'")

# 获取原生API的特征重要性
importance_gain = native_clf.get_score(importance_type='gain')  # gain, weight, cover
importance_gain = sorted(importance_gain.items(), key=lambda x: x[1], reverse=True)
print("\n原生API特征重要性 (按增益):")
for i, (feature, score) in enumerate(importance_gain[:5]):
    print(f"{i+1}. 特征 {feature}: {score:.4f}")

# =============================================================================
# 第5部分：超参数调优
# =============================================================================
print("\n第5部分：XGBoost - 超参数调优")
print("-" * 50)

print("使用网格搜索进行超参数调优")

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 初始化模型
xgb_grid = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 网格搜索
# 由于计算量大，这里我们只展示代码结构，不实际运行
"""
grid_search = GridSearchCV(
    estimator=xgb_grid,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 打印最佳参数
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
"""

# 使用随机搜索 (示例代码)
"""
from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}
xgb_random = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
random_search = RandomizedSearchCV(
    estimator=xgb_random,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
"""

# =============================================================================
# 第6部分：高级功能 - 早期停止和交叉验证
# =============================================================================
print("\n第6部分：XGBoost - 高级功能")
print("-" * 50)

# 早期停止（Early Stopping）
print("\n1. 早期停止(Early Stopping)")
print("早期停止可以防止过拟合，提前结束训练")

# 使用Scikit-Learn API的早期停止
xgb_early = xgb.XGBRegressor(
    n_estimators=1000,  # 设置大量的树
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

# 使用eval_set和early_stopping_rounds进行早期停止
xgb_early.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    early_stopping_rounds=10,  # 如果10轮内测试集性能没有提升，则停止
    verbose=False
)

print(f"最佳迭代次数: {xgb_early.best_iteration}")
print(f"最佳训练分数: {xgb_early.best_score:.4f}")

# 交叉验证
print("\n2. 交叉验证(Cross Validation)")
print("XGBoost内置交叉验证功能，便于超参数调优")

# 使用原生XGBoost API的交叉验证
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# 全部训练数据转换为DMatrix
dall = xgb.DMatrix(np.vstack((X_train, X_test)), label=np.hstack((y_train, y_test)))

# 进行交叉验证
# 1. params: 参数字典
# 2. dtrain: 训练数据
# 3. num_boost_round: 提升轮数
# 4. nfold: 交叉验证折数
# 5. stratified: 是否进行分层抽样
# 6. metrics: 评估指标列表
# 7. early_stopping_rounds: 早期停止轮数
# 8. seed: 随机种子
cv_results = xgb.cv(
    params,
    dall,
    num_boost_round=100,
    nfold=5,
    metrics={'rmse'},
    early_stopping_rounds=10,
    seed=42,
    verbose_eval=False
)

# 输出结果
print(f"交叉验证最佳RMSE: {cv_results['test-rmse-mean'].min():.4f} 在 {cv_results['test-rmse-mean'].argmin()+1} 轮")

# =============================================================================
# 第7部分：实际应用技巧
# =============================================================================
print("\n第7部分：XGBoost - 实际应用技巧")
print("-" * 50)

print("\n1. 处理缺失值")
print("XGBoost能自动处理缺失值，默认缺失值方向取决于数据")

print("\n2. 内存优化")
print("对于大数据集，可以使用以下优化：")
print(" - 使用DMatrix数据格式")
print(" - 使用external memory模式")
print(" - 设置process_type='update'")

print("\n3. 速度优化")
print(" - 使用tree_method='hist'加速训练")
print(" - GPU加速：tree_method='gpu_hist'")
print(" - 并行计算：nthread参数")

print("\n4. 正则化防止过拟合")
print(" - lambda: L2正则化项")
print(" - alpha: L1正则化项")
print(" - max_depth: 控制树深度")
print(" - gamma: 进行切分的最小损失减少")
print(" - min_child_weight: 最小子节点权重")
print(" - subsample & colsample_bytree: 随机样本和特征")

# =============================================================================
# 第8部分：模型解释与可视化
# =============================================================================
print("\n第8部分：XGBoost - 模型解释与可视化")
print("-" * 50)

# 可视化单棵树
print("\n1. 可视化单棵树")
print("通过可视化单棵树了解XGBoost的决策过程")

try:
    import graphviz
    # 保存一棵树的图形
    xgb.plot_tree(xgb_model, num_trees=0)
    plt.savefig('xgb_tree.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("决策树图已保存为 'xgb_tree.png'")
except ImportError:
    print("缺少graphviz包，无法可视化决策树")

# 重要特征交互
print("\n2. 特征交互")
print("了解哪些特征组合对预测最有影响")

# 使用scikit-learn API的决策树可视化
try:
    # 为了简单起见，我们仅可视化前两个特征
    from sklearn.tree import export_graphviz
    import pydot
    from io import StringIO
    
    estimator = xgb_model.get_booster().get_dump()[0]
    print(f"决策树结构示例 (前50个字符): {estimator[:50]}...")
except ImportError:
    print("缺少pydot包，无法可视化特征交互")

# =============================================================================
# 结语
# =============================================================================
print("\n结语")
print("-" * 50)
print("XGBoost是一个功能强大的梯度提升框架，适用于各种机器学习任务。")
print("通过本示例，您已经了解了XGBoost的基本用法、超参数调优、高级功能和实际应用技巧。")
print("更多信息请查阅官方文档: https://xgboost.readthedocs.io/")
print("\n祝您使用愉快！")
print("=" * 80) 