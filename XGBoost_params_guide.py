#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
XGBoost 参数详解与DMatrix使用指南

本文件详细介绍了XGBoost中XGBClassifier和XGBRegressor的参数，
以及DMatrix数据结构的使用方法。适合作为XGBoost_demo.py的补充材料。

作者：AI助手
日期：2023年
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置随机种子，确保结果可重复
np.random.seed(42)

print("=" * 80)
print("XGBoost 参数详解与DMatrix使用指南")
print("=" * 80)

# =============================================================================
# 第1部分：XGBClassifier和XGBRegressor的参数详解
# =============================================================================
print("\n第1部分：XGBClassifier和XGBRegressor的参数详解")
print("-" * 50)

print("""
XGBClassifier和XGBRegressor的参数可以分为几个主要类别：

一、基本参数：控制总体训练行为
二、树参数：控制每棵树的生成
三、学习率参数：控制训练过程
四、正则化参数：控制模型复杂度
五、其他参数：其他高级选项
""")

print("\n一、基本参数")
print("-" * 30)
params_basic = {
    "objective": {
        "描述": "定义学习任务和相应的学习目标",
        "常用选项": {
            "reg:squarederror": "回归问题的平方误差",
            "binary:logistic": "二分类的逻辑回归，输出概率",
            "binary:hinge": "二分类的铰链损失，输出0或1",
            "multi:softmax": "多分类问题，使用softmax，需要设置num_class",
            "multi:softprob": "多分类问题，输出概率，需要设置num_class",
            "rank:pairwise": "排序问题的配对损失"
        },
        "默认值": "reg:squarederror (XGBRegressor) / binary:logistic (XGBClassifier)"
    },
    "booster": {
        "描述": "使用哪种基学习器",
        "常用选项": {
            "gbtree": "基于树的模型",
            "gblinear": "线性模型",
            "dart": "基于树的模型，使用Dropout技术"
        },
        "默认值": "gbtree"
    },
    "verbosity": {
        "描述": "打印消息的详细程度",
        "常用选项": {
            "0": "静默模式",
            "1": "警告模式",
            "2": "信息模式",
            "3": "调试模式"
        },
        "默认值": "1"
    },
    "n_estimators": {
        "描述": "提升轮数，即生成的树的数量",
        "默认值": "100"
    },
    "n_jobs": {
        "描述": "并行线程数。设为-1使用所有CPU",
        "默认值": "1"
    },
    "random_state": {
        "描述": "随机种子",
        "默认值": "None"
    }
}

# 打印基本参数
for param, details in params_basic.items():
    print(f"\n参数: {param}")
    print(f"  描述: {details['描述']}")
    if '常用选项' in details:
        print("  常用选项:")
        for option, desc in details['常用选项'].items():
            print(f"    - {option}: {desc}")
    print(f"  默认值: {details['默认值']}")

print("\n二、树参数")
print("-" * 30)
params_tree = {
    "max_depth": {
        "描述": "树的最大深度。增加此值使模型更复杂，可能导致过拟合",
        "范围": "通常在3-10之间",
        "默认值": "6"
    },
    "min_child_weight": {
        "描述": "子节点中所需的最小样本权重和。较大的值防止过拟合",
        "范围": "通常在1-10之间",
        "默认值": "1"
    },
    "max_delta_step": {
        "描述": "每棵树的权重估计的最大delta步长。通常不需要设置",
        "范围": "通常在0-10之间",
        "默认值": "0"
    },
    "subsample": {
        "描述": "训练实例的子样本比例。较低的值防止过拟合",
        "范围": "典型值为0.5-1.0",
        "默认值": "1.0"
    },
    "colsample_bytree": {
        "描述": "构建每棵树时列的子样本比例",
        "范围": "典型值为0.5-1.0",
        "默认值": "1.0"
    },
    "colsample_bylevel": {
        "描述": "树的每一级的列子样本比例",
        "范围": "典型值为0.5-1.0",
        "默认值": "1.0"
    },
    "colsample_bynode": {
        "描述": "每个节点的列子样本比例",
        "范围": "典型值为0.5-1.0",
        "默认值": "1.0"
    },
    "grow_policy": {
        "描述": "控制新节点如何添加到树中",
        "常用选项": {
            "depthwise": "从树的顶部到底部分层增长",
            "lossguide": "基于损失减少选择最佳节点扩展"
        },
        "默认值": "depthwise"
    },
    "tree_method": {
        "描述": "树构建算法",
        "常用选项": {
            "auto": "启发式选择最快的方法",
            "exact": "精确贪婪算法",
            "approx": "近似贪婪算法",
            "hist": "更快的近似算法，使用直方图计数",
            "gpu_hist": "GPU版本的hist算法"
        },
        "默认值": "auto"
    }
}

# 打印树参数
for param, details in params_tree.items():
    print(f"\n参数: {param}")
    print(f"  描述: {details['描述']}")
    if '常用选项' in details:
        print("  常用选项:")
        for option, desc in details['常用选项'].items():
            print(f"    - {option}: {desc}")
    if '范围' in details:
        print(f"  范围: {details['范围']}")
    print(f"  默认值: {details['默认值']}")

print("\n三、学习率参数")
print("-" * 30)
params_lr = {
    "learning_rate": {
        "描述": "学习率/步长/收缩因子。每步收缩特征权重以防止过拟合",
        "别名": "eta",
        "范围": "典型值为0.01-0.3",
        "默认值": "0.3"
    },
    "gamma": {
        "描述": "在节点拆分时所需的最小损失减少。值越大，算法越保守",
        "范围": "典型值为0-10",
        "默认值": "0"
    },
    "max_leaves": {
        "描述": "最大叶节点数（仅在lossguide生长策略时有效）",
        "默认值": "0 (无限制)"
    }
}

# 打印学习率参数
for param, details in params_lr.items():
    print(f"\n参数: {param}")
    print(f"  描述: {details['描述']}")
    if '别名' in details:
        print(f"  别名: {details['别名']}")
    if '范围' in details:
        print(f"  范围: {details['范围']}")
    print(f"  默认值: {details['默认值']}")

print("\n四、正则化参数")
print("-" * 30)
params_reg = {
    "lambda": {
        "描述": "L2正则化项。增加此值使模型更保守",
        "别名": "reg_lambda",
        "范围": "典型值为0.1-10",
        "默认值": "1"
    },
    "alpha": {
        "描述": "L1正则化项。增加此值使模型更保守，导致更多权重为零（特征选择）",
        "别名": "reg_alpha",
        "范围": "典型值为0-10",
        "默认值": "0"
    },
    "scale_pos_weight": {
        "描述": "控制正负权重的平衡。对不平衡数据有用",
        "用途": "对于不平衡分类，设置为负类数/正类数",
        "默认值": "1"
    }
}

# 打印正则化参数
for param, details in params_reg.items():
    print(f"\n参数: {param}")
    print(f"  描述: {details['描述']}")
    if '别名' in details:
        print(f"  别名: {details['别名']}")
    if '范围' in details:
        print(f"  范围: {details['范围']}")
    if '用途' in details:
        print(f"  用途: {details['用途']}")
    print(f"  默认值: {details['默认值']}")

print("\n五、其他参数")
print("-" * 30)
params_other = {
    "early_stopping_rounds": {
        "描述": "如果在指定轮数内没有改善，则停止训练",
        "用法": "需要设置validation_set或eval_set，通常设为10-50",
        "默认值": "None (不早停)"
    },
    "num_class": {
        "描述": "多分类问题的类别数量",
        "适用于": "多分类问题 (objective='multi:softmax'或'multi:softprob')",
        "默认值": "None"
    },
    "importance_type": {
        "描述": "特征重要性的类型",
        "常用选项": {
            "weight": "特征在所有树中出现的次数",
            "gain": "特征对模型的贡献度 (推荐)",
            "cover": "特征覆盖的相对数量"
        },
        "默认值": "weight"
    },
    "eval_metric": {
        "描述": "验证数据的评估指标",
        "常用选项": {
            "rmse": "均方根误差 (回归)",
            "mae": "平均绝对误差 (回归)",
            "logloss": "负对数似然 (分类)",
            "error": "分类错误率 (分类)",
            "auc": "曲线下面积 (分类)",
            "ndcg": "归一化折损累积增益 (排序)"
        },
        "默认值": "取决于objective参数"
    }
}

# 打印其他参数
for param, details in params_other.items():
    print(f"\n参数: {param}")
    print(f"  描述: {details['描述']}")
    if '常用选项' in details:
        print("  常用选项:")
        for option, desc in details['常用选项'].items():
            print(f"    - {option}: {desc}")
    if '用法' in details:
        print(f"  用法: {details['用法']}")
    if '适用于' in details:
        print(f"  适用于: {details['适用于']}")
    print(f"  默认值: {details['默认值']}")

# =============================================================================
# 第2部分：XGBoost的DMatrix详细介绍
# =============================================================================
print("\n第2部分：XGBoost的DMatrix详细介绍")
print("-" * 50)

print("""
DMatrix是XGBoost的专用数据结构，优点包括：
1. 性能高效：针对XGBoost算法优化的存储结构
2. 内存高效：特别是处理大数据集时
3. 可缓存特征：避免重复计算
4. 支持缺失值处理：自动识别和处理

以下将介绍DMatrix的创建方法、主要参数和常用操作。
""")

# 创建一些演示数据
print("\n创建示例数据用于DMatrix演示...")
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDMatrix的创建方法：")
print("-" * 30)

print("""
1. 从NumPy数组创建DMatrix
""")
print("代码示例：")
print("dtrain = xgb.DMatrix(X_train, label=y_train)")
print("dtest = xgb.DMatrix(X_test, label=y_test)")

# 创建DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print(f"\n创建的DMatrix信息：")
print(f"训练集DMatrix维度: {dtrain.num_row()} 行 x {dtrain.num_col()} 列")
print(f"测试集DMatrix维度: {dtest.num_row()} 行 x {dtest.num_col()} 列")

print("""
2. 从Pandas DataFrame创建DMatrix
""")
print("代码示例：")
print("df_train = pd.DataFrame(X_train, columns=[f'f{i}' for i in range(X_train.shape[1])])")
print("df_train['label'] = y_train")
print("dtrain = xgb.DMatrix(df_train.drop('label', axis=1), label=df_train['label'])")

# 创建DataFrame
df_train = pd.DataFrame(X_train, columns=[f'f{i}' for i in range(X_train.shape[1])])
df_train['label'] = y_train

print("""
3. 从稀疏矩阵创建DMatrix (适用于高维稀疏特征)
""")
print("代码示例：")
print("from scipy.sparse import csr_matrix")
print("X_sparse = csr_matrix(X_train)")
print("dtrain_sparse = xgb.DMatrix(X_sparse, label=y_train)")

print("""
4. 从文件加载DMatrix
""")
print("代码示例：")
print("# 先保存数据到XGBoost格式")
print("dtrain.save_binary('dtrain.buffer')")
print("# 从文件加载")
print("dtrain_loaded = xgb.DMatrix('dtrain.buffer')")

# 保存DMatrix到文件
dtrain.save_binary('dtrain.buffer')

print("\nDMatrix的主要参数：")
print("-" * 30)
dmatrix_params = {
    "data": {
        "描述": "特征数据，可以是NumPy数组、SciPy稀疏矩阵、Pandas DataFrame、文件路径等",
        "必需": "是"
    },
    "label": {
        "描述": "标签/目标变量，一维数组",
        "必需": "否，但训练时通常需要提供"
    },
    "weight": {
        "描述": "每个实例的权重，一维数组",
        "必需": "否",
        "用途": "对某些样本赋予更高重要性"
    },
    "base_margin": {
        "描述": "每个实例的初始预测，一维数组",
        "必需": "否",
        "用途": "用于增量训练或已知偏差调整"
    },
    "missing": {
        "描述": "指定缺失值的表示",
        "默认值": "np.nan",
        "用法": "如果数据中使用特定值表示缺失，如-999，可设置missing=-999"
    },
    "silent": {
        "描述": "是否抑制打印消息",
        "默认值": "True",
        "注意": "已弃用，使用verbosity代替"
    },
    "feature_names": {
        "描述": "特征名称列表",
        "必需": "否",
        "用途": "提高可解释性，用于特征重要性等"
    },
    "feature_types": {
        "描述": "每个特征的类型，可以是'q'(数值)或'c'(分类)",
        "必需": "否",
        "默认值": "自动推断"
    },
    "nthread": {
        "描述": "使用的线程数",
        "默认值": "None (使用系统默认值)",
        "注意": "通常不需要设置，会使用XGBoost全局配置"
    }
}

# 打印DMatrix参数
for param, details in dmatrix_params.items():
    print(f"\n参数: {param}")
    print(f"  描述: {details['描述']}")
    if '必需' in details:
        print(f"  必需: {details['必需']}")
    if '默认值' in details:
        print(f"  默认值: {details['默认值']}")
    if '用途' in details:
        print(f"  用途: {details['用途']}")
    if '用法' in details:
        print(f"  用法: {details['用法']}")
    if '注意' in details:
        print(f"  注意: {details['注意']}")

print("\nDMatrix的常用操作：")
print("-" * 30)

print("""
1. 设置权重
""")
print("代码示例：")
print("weights = np.ones(len(y_train))")
print("weights[y_train == 1] = 2  # 给正类更高的权重")
print("dtrain.set_weight(weights)")

# 设置权重示例
weights = np.ones(len(y_train))
weights[y_train == 1] = 2  # 给正类更高的权重
dtrain.set_weight(weights)

print("""
2. 设置特征名称
""")
print("代码示例：")
print("feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]")
print("dtrain.feature_names = feature_names")

# 设置特征名称
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
try:
    dtrain.feature_names = feature_names
except AttributeError:
    # 较新版本可能不支持直接设置
    print("注意：较新版本需要在创建时设置：dtrain = xgb.DMatrix(data, feature_names=feature_names)")

print("""
3. 获取标签
""")
print("代码示例：")
print("labels = dtrain.get_label()")
print(f"标签分布统计：正例 {sum(dtrain.get_label() == 1)}，负例 {sum(dtrain.get_label() == 0)}")

# 获取标签
labels = dtrain.get_label()
print(f"标签分布统计：正例 {sum(labels == 1)}，负例 {sum(labels == 0)}")

print("""
4. 获取权重
""")
print("代码示例：")
print("weights = dtrain.get_weight()")

# 获取权重
weights = dtrain.get_weight()
print(f"权重平均值: {weights.mean():.4f}")

print("""
5. 获取DMatrix的形状
""")
print("代码示例：")
print("num_rows = dtrain.num_row()")
print("num_cols = dtrain.num_col()")

print(f"DMatrix形状: {dtrain.num_row()} 行 x {dtrain.num_col()} 列")

print("""
6. 从多个来源创建DMatrix（高级用法）
""")
print("代码示例：")
print("# 假设有多个数据源")
print("X1 = X_train[:500]")
print("y1 = y_train[:500]")
print("X2 = X_train[500:]")
print("y2 = y_train[500:]")
print("# 先创建两个DMatrix")
print("dtrain1 = xgb.DMatrix(X1, label=y1)")
print("dtrain2 = xgb.DMatrix(X2, label=y2)")
print("# 合并训练")
print("model = xgb.train(params, dtrain1)")
print("# 使用更新模式训练")
print("model = xgb.train(params, dtrain2, xgb_model=model)")

# 示例：高级数据处理
print("\n高级用法：处理缺失值和分类特征")
print("-" * 30)

print("""
1. 处理缺失值
XGBoost能够自动处理NaN值为缺失值，不需要额外插值。
""")
print("代码示例：")
print("X_with_missing = X_train.copy()")
print("# 随机引入一些缺失值")
print("mask = np.random.rand(*X_with_missing.shape) < 0.1")
print("X_with_missing[mask] = np.nan")
print("dtrain_missing = xgb.DMatrix(X_with_missing, label=y_train)")

# 代码示例：引入缺失值
X_with_missing = X_train.copy()
mask = np.random.rand(*X_with_missing.shape) < 0.1
X_with_missing[mask] = np.nan
print(f"引入的缺失值数量: {np.isnan(X_with_missing).sum()}")

print("""
2. 自定义缺失值表示
如果你的数据使用特定值表示缺失（如-999），可以在创建DMatrix时指定。
""")
print("代码示例：")
print("X_custom_missing = X_train.copy()")
print("X_custom_missing[mask] = -999")
print("dtrain_custom = xgb.DMatrix(X_custom_missing, label=y_train, missing=-999)")

print("""
3. 使用DMatrix处理分类特征
XGBoost倾向于使用独热编码处理分类特征。
""")
print("代码示例：")
print("# 假设我们有一列分类特征")
print("cat_feature = np.random.randint(0, 5, size=len(X_train))")
print("# 需要先进行独热编码")
print("from sklearn.preprocessing import OneHotEncoder")
print("encoder = OneHotEncoder(sparse=False)")
print("cat_encoded = encoder.fit_transform(cat_feature.reshape(-1, 1))")
print("# 合并到特征矩阵")
print("X_with_cat = np.hstack([X_train, cat_encoded])")
print("dtrain_cat = xgb.DMatrix(X_with_cat, label=y_train)")

# 示例：DMatrix与训练流程集成
print("\nDMatrix与训练流程集成")
print("-" * 30)

print("""
使用DMatrix的完整训练流程示例：
""")
print("代码示例：")
print("""
# 1. 创建DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 2. 设置参数
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': ['error', 'logloss', 'auc']
}

# 3. 创建watchlist（用于监控训练过程）
watchlist = [(dtrain, 'train'), (dtest, 'test')]

# 4. 训练模型
num_rounds = 100
evals_result = {}  # 存储评估结果
model = xgb.train(
    params, 
    dtrain, 
    num_rounds, 
    evals=watchlist,
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=True  # 打印评估信息
)

# 5. 获取最佳迭代次数
best_iteration = model.best_iteration
print(f"最佳迭代次数: {best_iteration}")

# 6. 可视化训练过程
epochs = len(evals_result['train']['error'])
x_axis = range(epochs)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, evals_result['train']['error'], label='Train')
plt.plot(x_axis, evals_result['test']['error'], label='Test')
plt.legend()
plt.xlabel('Rounds')
plt.ylabel('Classification Error')
plt.title('XGBoost Error')

plt.subplot(1, 2, 2)
plt.plot(x_axis, evals_result['train']['auc'], label='Train')
plt.plot(x_axis, evals_result['test']['auc'], label='Test')
plt.legend()
plt.xlabel('Rounds')
plt.ylabel('AUC')
plt.title('XGBoost AUC')

plt.tight_layout()
plt.savefig('xgb_training_metrics.png', dpi=300, bbox_inches='tight')

# 7. 预测
pred_proba = model.predict(dtest)
pred_class = (pred_proba > 0.5).astype(int)

# 8. 保存模型
model.save_model('xgb_model.json')

# 9. 加载模型
loaded_model = xgb.Booster()
loaded_model.load_model('xgb_model.json')
""")

# 进行实际的简单训练演示
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': ['error', 'logloss']
}

print("\n执行简短的训练示例...")
# 创建watchlist
watchlist = [(dtrain, 'train'), (dtest, 'test')]

# 训练轮数少一些，只是为了演示
num_rounds = 10
evals_result = {}

# 训练模型
model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evals=watchlist,
    evals_result=evals_result,
    verbose_eval=False
)

print("\n训练结果摘要：")
print(f"最终训练误差: {evals_result['train']['error'][-1]:.4f}")
print(f"最终测试误差: {evals_result['test']['error'][-1]:.4f}")

# =============================================================================
# 结语
# =============================================================================
print("\n结语")
print("-" * 50)
print("本指南详细介绍了XGBoost中XGBClassifier和XGBRegressor的主要参数以及DMatrix的使用方法。")
print("有效的参数调优和正确使用DMatrix是充分利用XGBoost性能的关键。")
print("更多信息请参考官方文档: https://xgboost.readthedocs.io/")
print("\n祝您使用愉快！")
print("=" * 80) 