#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SHAP (SHapley Additive exPlanations) 详细使用指南与示例

SHAP是一个用于解释机器学习模型的强大工具，它基于博弈论的Shapley值概念，
为每个特征分配重要性值，解释模型如何做出特定预测。SHAP提供了一致、局部准确
且能处理缺失数据的解释方法，被认为是当前最先进的模型解释工具之一。

本文件详细介绍SHAP库的使用方法，包括：
1. SHAP值的基本概念
2. 不同模型的SHAP解释器
3. 各种可视化方法
4. 全局和局部解释
5. 与各种ML框架的集成

SHAP主要优势：
- 理论基础扎实：基于博弈论的Shapley值
- 全局和局部解释：既可解释整体模型也可解释单个预测
- 模型无关性：支持各种ML模型
- 丰富的可视化工具：直观展示特征对预测的影响

作者：AI助手
日期：2023年
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
import shap
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可重复
np.random.seed(42)

print("=" * 80)
print("SHAP (SHapley Additive exPlanations) 详细使用指南与示例")
print("=" * 80)

# =============================================================================
# 第1部分：SHAP基本概念
# =============================================================================
print("\n第1部分：SHAP基本概念")
print("-" * 50)

print("""
SHAP值是基于博弈论中的Shapley值。关键概念:

1. 每个特征都是"玩家"，预测是"收益"
2. SHAP值解释特征如何从基线值(平均预测)推动模型输出
3. 正的SHAP值表示特征增加预测值，负的表示减少
4. SHAP值的总和等于实际预测减去平均预测

SHAP提供了多种解释器，适用于不同模型:
- TreeExplainer: 优化的树模型解释器(XGBoost, LightGBM, RandomForest等)
- DeepExplainer: 深度学习模型解释器
- KernelExplainer: 通用解释器，适用于任何模型，但计算较慢
- LinearExplainer: 线性模型解释器
- GradientExplainer: 基于梯度的解释器
""")

# =============================================================================
# 第2部分：准备数据和模型
# =============================================================================
print("\n第2部分：准备数据和模型")
print("-" * 50)

# 为解释做准备，我们需要数据和一个训练好的模型
print("加载和准备数据...")

# 加载回归数据集(波士顿房价)
try:
    # sklearn 1.0+版本已弃用load_boston
    boston = load_boston()
    X_reg, y_reg = boston.data, boston.target
    feature_names_reg = boston.feature_names
except:
    # 如果load_boston不可用，使用模拟数据
    print("注意: 使用模拟的回归数据集(sklearn已弃用波士顿房价数据集)")
    from sklearn.datasets import make_regression
    X_reg, y_reg = make_regression(n_samples=506, n_features=13, noise=10, random_state=42)
    feature_names_reg = [f'feature_{i}' for i in range(X_reg.shape[1])]

# 加载分类数据集(乳腺癌)
cancer = load_breast_cancer()
X_cls, y_cls = cancer.data, cancer.target
feature_names_cls = cancer.feature_names

# 分割数据为训练集和测试集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# 训练回归模型 - 使用XGBoost
print("训练回归模型(XGBoost)...")
xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_reg.fit(X_train_reg, y_train_reg)

# 训练分类模型 - 使用XGBoost
print("训练分类模型(XGBoost)...")
xgb_cls = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_cls.fit(X_train_cls, y_train_cls)

# 训练另一个回归模型 - 使用RandomForest(用于比较不同模型)
print("训练回归模型(RandomForest)...")
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

print("模型训练完成，准备开始SHAP分析...")

# =============================================================================
# 第3部分：TreeExplainer - 解释树模型
# =============================================================================
print("\n第3部分：TreeExplainer - 解释树模型")
print("-" * 50)

print("TreeExplainer是针对树模型优化的解释器，速度最快，支持各种树模型")

# 创建TreeExplainer实例
# TreeExplainer为XGBoost, LightGBM, CatBoost, sklearn的树模型优化
# 参数model是要解释的模型
print("\n创建XGBoost回归模型的SHAP解释器...")
explainer_xgb_reg = shap.TreeExplainer(xgb_reg)

# 计算测试集的SHAP值
# TreeExplainer.shap_values()计算SHAP值
# 返回一个形状为(样本数, 特征数)的数组
print("计算测试集SHAP值...")
shap_values_xgb_reg = explainer_xgb_reg.shap_values(X_test_reg)

# 打印第一个样本的SHAP值
print("\nXGBoost回归模型第一个测试样本的SHAP值（前5个特征）:")
for i in range(min(5, len(feature_names_reg))):
    print(f"{feature_names_reg[i]}: {shap_values_xgb_reg[0][i]:.4f}")

# 计算预期值(base value) - 模型输出的平均值
print(f"\n预期值(base value): {explainer_xgb_reg.expected_value:.4f}")
print("SHAP值是相对于这个基准值的贡献")

# 创建分类模型的解释器
print("\n创建XGBoost分类模型的SHAP解释器...")
explainer_xgb_cls = shap.TreeExplainer(xgb_cls)

# 计算分类模型的SHAP值
# 对于二分类，返回一个数组表示正类的贡献
# 对于多分类，返回一个列表，每个元素是一个类别的SHAP值
shap_values_xgb_cls = explainer_xgb_cls.shap_values(X_test_cls)

# 打印分类模型的预期值
if isinstance(explainer_xgb_cls.expected_value, list):
    # 多分类
    print(f"\n各类别的预期值: {[float(f'{v:.4f}') for v in explainer_xgb_cls.expected_value]}")
else:
    # 二分类
    print(f"\n预期值: {explainer_xgb_cls.expected_value:.4f}")

# =============================================================================
# 第4部分：可视化SHAP值 - 概要图(Summary Plot)
# =============================================================================
print("\n第4部分：可视化SHAP值 - 概要图(Summary Plot)")
print("-" * 50)

print("""
概要图展示了特征如何影响模型输出:
- 每个点代表一个样本的一个特征
- x轴表示SHAP值(特征对预测的贡献)
- 颜色表示特征值(红色高，蓝色低)
- 点的分布展示了特征影响的分布
- 特征按平均绝对SHAP值排序
""")

# 生成概要图 - 回归模型
print("\n生成XGBoost回归模型的SHAP概要图...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_xgb_reg, X_test_reg, feature_names=feature_names_reg, show=False)
plt.title("XGBoost回归模型的SHAP概要图")
plt.tight_layout()
plt.savefig('shap_summary_xgb_reg.png', dpi=300, bbox_inches='tight')
plt.close()
print("概要图已保存为 'shap_summary_xgb_reg.png'")

# 生成概要图 - 分类模型
print("\n生成XGBoost分类模型的SHAP概要图...")
plt.figure(figsize=(12, 8))
# 对于二分类，我们通常只关注正类的SHAP值
if isinstance(shap_values_xgb_cls, list):
    # 当shap_values_xgb_cls是一个列表时，表示是多分类
    shap.summary_plot(shap_values_xgb_cls[1], X_test_cls, feature_names=feature_names_cls, show=False)
else:
    # 否则是二分类
    shap.summary_plot(shap_values_xgb_cls, X_test_cls, feature_names=feature_names_cls, show=False)
plt.title("XGBoost分类模型的SHAP概要图")
plt.tight_layout()
plt.savefig('shap_summary_xgb_cls.png', dpi=300, bbox_inches='tight')
plt.close()
print("概要图已保存为 'shap_summary_xgb_cls.png'")

# 生成条形图概要图
print("\n生成条形图概要图(特征重要性)...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_xgb_reg, X_test_reg, feature_names=feature_names_reg, plot_type="bar", show=False)
plt.title("XGBoost回归模型的SHAP特征重要性")
plt.tight_layout()
plt.savefig('shap_bar_xgb_reg.png', dpi=300, bbox_inches='tight')
plt.close()
print("条形图概要图已保存为 'shap_bar_xgb_reg.png'")

# =============================================================================
# 第5部分：SHAP依赖图(Dependence Plot)
# =============================================================================
print("\n第5部分：SHAP依赖图(Dependence Plot)")
print("-" * 50)

print("""
依赖图展示了特征值与其SHAP值的关系:
- 每个点是一个样本
- x轴是特征值
- y轴是SHAP值
- 颜色表示另一个交互特征的值(热力色)
- 这帮助理解特征如何影响预测，以及特征间的交互
""")

# 生成依赖图
# 我们选择最重要的特征(SHAP值绝对值最大的特征)
if X_test_reg.shape[1] > 0:
    most_important_feature_idx = np.argmax(np.abs(shap_values_xgb_reg).mean(axis=0))
    most_important_feature = feature_names_reg[most_important_feature_idx]
    
    print(f"\n生成特征 '{most_important_feature}' 的依赖图...")
    plt.figure(figsize=(12, 8))
    shap.dependence_plot(most_important_feature_idx, shap_values_xgb_reg, X_test_reg, 
                        feature_names=feature_names_reg, show=False)
    plt.title(f"特征 '{most_important_feature}' 的SHAP依赖图")
    plt.tight_layout()
    plt.savefig('shap_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("依赖图已保存为 'shap_dependence.png'")
    
    # 显式指定交互特征的依赖图
    # 找到第二重要的特征
    second_important_feature_idx = np.argsort(np.abs(shap_values_xgb_reg).mean(axis=0))[-2]
    second_important_feature = feature_names_reg[second_important_feature_idx]
    
    print(f"\n生成特征 '{most_important_feature}' 与 '{second_important_feature}' 的交互依赖图...")
    plt.figure(figsize=(12, 8))
    shap.dependence_plot(most_important_feature_idx, shap_values_xgb_reg, X_test_reg, 
                        feature_names=feature_names_reg, interaction_index=second_important_feature_idx, 
                        show=False)
    plt.title(f"特征 '{most_important_feature}' 与 '{second_important_feature}' 的SHAP交互依赖图")
    plt.tight_layout()
    plt.savefig('shap_interaction_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("交互依赖图已保存为 'shap_interaction_dependence.png'")

# =============================================================================
# 第6部分：SHAP力图(Force Plot)
# =============================================================================
print("\n第6部分：SHAP力图(Force Plot)")
print("-" * 50)

print("""
力图展示单个预测的解释:
- 红色箭头向右推动预测值增加
- 蓝色箭头向左推动预测值减少
- 箭头长度表示SHAP值的大小
- 起始点是预期值(base value)
- 终点是实际预测值
""")

# 单个预测的力图
print("\n生成单个预测的力图...")
plt.figure(figsize=(16, 3))
shap.initjs()  # 初始化JavaScript可视化

# 在matplotlib中渲染力图
if X_test_reg.shape[0] > 0:
    shap.force_plot(explainer_xgb_reg.expected_value, shap_values_xgb_reg[0,:], 
                    X_test_reg[0,:], feature_names=feature_names_reg, 
                    matplotlib=True, show=False)
    plt.title("单个预测的SHAP力图")
    plt.tight_layout()
    plt.savefig('shap_force_single.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("单个预测的力图已保存为 'shap_force_single.png'")

# 多个预测的力图
# 注意：这通常生成交互式HTML，在静态图像中效果有限
print("\n生成多个预测的力图(仅显示几个样本)...")
num_samples = min(10, X_test_reg.shape[0])
plt.figure(figsize=(16, 6))
shap.force_plot(explainer_xgb_reg.expected_value, shap_values_xgb_reg[:num_samples,:], 
               X_test_reg[:num_samples,:], feature_names=feature_names_reg,
               matplotlib=True, show=False)
plt.title("多个预测的SHAP力图")
plt.tight_layout()
plt.savefig('shap_force_multiple.png', dpi=300, bbox_inches='tight')
plt.close()
print("多个预测的力图已保存为 'shap_force_multiple.png'")

# =============================================================================
# 第7部分：SHAP决策图(Decision Plot)
# =============================================================================
print("\n第7部分：SHAP决策图(Decision Plot)")
print("-" * 50)

print("""
决策图展示了从基准到最终预测的路径:
- y轴是特征
- x轴是累积SHAP值
- 从底部(预期值)开始，到顶部(实际预测)结束
- 每个特征的贡献按其SHAP值显示
- 非常适合理解复杂模型的决策过程
""")

# 创建决策图
print("\n生成决策图...")
# 选择几个样本进行展示
num_samples = min(10, X_test_reg.shape[0])
plt.figure(figsize=(12, 10))
try:
    # 需要SHAP库新版本
    shap.decision_plot(explainer_xgb_reg.expected_value, shap_values_xgb_reg[:num_samples,:], 
                      feature_names=feature_names_reg, show=False)
    plt.title("SHAP决策图")
    plt.tight_layout()
    plt.savefig('shap_decision.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("决策图已保存为 'shap_decision.png'")
except AttributeError:
    print("您的SHAP库版本可能不支持决策图。请升级到最新版本: pip install --upgrade shap")

# =============================================================================
# 第8部分：SHAP瀑布图(Waterfall Plot)
# =============================================================================
print("\n第8部分：SHAP瀑布图(Waterfall Plot)")
print("-" * 50)

print("""
瀑布图详细展示单个预测的贡献:
- 从预期值开始，展示每个特征如何推动预测
- 特征按贡献绝对值排序
- 红色表示正向贡献，蓝色表示负向贡献
- 最后一行是最终预测值
""")

# 创建瀑布图
print("\n生成瀑布图...")
if X_test_reg.shape[0] > 0:
    try:
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap.Explanation(values=shap_values_xgb_reg[0], 
                                             base_values=explainer_xgb_reg.expected_value, 
                                             data=X_test_reg[0],
                                             feature_names=feature_names_reg), show=False)
        plt.title("SHAP瀑布图")
        plt.tight_layout()
        plt.savefig('shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("瀑布图已保存为 'shap_waterfall.png'")
    except (AttributeError, ValueError) as e:
        print(f"创建瀑布图时出错，可能需要更新SHAP库: {e}")

# =============================================================================
# 第9部分：SHAP与其他模型 - KernelExplainer
# =============================================================================
print("\n第9部分：SHAP与其他模型 - KernelExplainer")
print("-" * 50)

print("""
KernelExplainer是一种模型无关的解释器:
- 基于LIME的加权局部线性回归
- 可以解释任何模型，无论黑盒如何
- 计算速度较慢
- 适用于无法使用专用解释器的模型
""")

# 创建一个简单的黑盒函数作为示例
print("\n创建一个示例黑盒函数...")
def black_box_function(X):
    """示例黑盒函数，可以是任何模型的预测函数"""
    return rf_reg.predict(X)

# 创建KernelExplainer
print("创建KernelExplainer...")
# 使用一部分数据作为背景数据集
# 背景数据集用于估计特征的预期效果
background = X_train_reg[:100]  # 使用100个样本作为背景
kernel_explainer = shap.KernelExplainer(black_box_function, background)

# 计算SHAP值
print("计算少量样本的SHAP值(KernelExplainer计算较慢)...")
# 只计算几个样本，因为KernelExplainer计算很慢
X_explain = X_test_reg[:10]
kernel_shap_values = kernel_explainer.shap_values(X_explain)

# 创建概要图
print("\n生成KernelExplainer的SHAP概要图...")
plt.figure(figsize=(12, 8))
shap.summary_plot(kernel_shap_values, X_explain, feature_names=feature_names_reg, show=False)
plt.title("KernelExplainer的SHAP概要图")
plt.tight_layout()
plt.savefig('shap_kernel_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("KernelExplainer概要图已保存为 'shap_kernel_summary.png'")

# =============================================================================
# 第10部分：集成SHAP到机器学习流程
# =============================================================================
print("\n第10部分：集成SHAP到机器学习流程")
print("-" * 50)

print("""
SHAP可以集成到机器学习流程的多个阶段:

1. 模型解释:
   - 理解模型如何做出预测
   - 验证模型是否使用了合理的特征
   - 检测潜在的偏见或公平性问题

2. 特征选择:
   - 使用SHAP值筛选最重要特征
   - 移除低重要性特征简化模型

3. 模型调试:
   - 发现模型的错误模式
   - 了解模型何时可能失效

4. 模型监控:
   - 跟踪模型解释随时间变化
   - 检测数据漂移
""")

# 演示基于SHAP值的特征选择
print("\n演示基于SHAP值的特征选择:")
# 计算每个特征的平均绝对SHAP值
feature_importance = np.abs(shap_values_xgb_reg).mean(axis=0)
# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names_reg,
    'importance': feature_importance
})
importance_df = importance_df.sort_values('importance', ascending=False)

# 打印最重要的5个特征
print("\n基于SHAP值的前5个重要特征:")
for i, (feature, importance) in enumerate(zip(importance_df['feature'][:5], importance_df['importance'][:5])):
    print(f"{i+1}. {feature}: {importance:.4f}")

# 使用top N特征训练新模型
print("\n使用最重要的特征子集训练新模型...")
top_n = min(5, len(feature_names_reg))  # 选择前5个特征
top_features = importance_df['feature'][:top_n].values
# 获取这些特征的索引
top_indices = [list(feature_names_reg).index(feature) for feature in top_features]
# 创建只包含这些特征的数据集
X_train_top = X_train_reg[:, top_indices]
X_test_top = X_test_reg[:, top_indices]

# 训练新模型
print(f"训练仅使用前{top_n}个重要特征的模型...")
xgb_top = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_top.fit(X_train_top, y_train_reg)

# 比较性能
from sklearn.metrics import mean_squared_error
y_pred_full = xgb_reg.predict(X_test_reg)
y_pred_top = xgb_top.predict(X_test_top)
rmse_full = np.sqrt(mean_squared_error(y_test_reg, y_pred_full))
rmse_top = np.sqrt(mean_squared_error(y_test_reg, y_pred_top))

print(f"\n全特征模型RMSE: {rmse_full:.4f}")
print(f"仅使用前{top_n}个特征的模型RMSE: {rmse_top:.4f}")
print(f"性能变化: {(rmse_top-rmse_full)/rmse_full*100:.2f}%")

# =============================================================================
# 第11部分：高级SHAP应用
# =============================================================================
print("\n第11部分：高级SHAP应用")
print("-" * 50)

print("""
高级SHAP应用示例:

1. 聚类分析与SHAP:
   - 对相似的解释进行聚类
   - 发现数据中的自然分组

2. 交互式仪表盘:
   - 创建实时SHAP解释的仪表盘
   - 允许用户探索模型决策

3. 组合模型的SHAP:
   - 比较多个模型的SHAP值
   - 组合多个模型的解释
""")

# =============================================================================
# 结语
# =============================================================================
print("\n结语")
print("-" * 50)
print("SHAP是解释机器学习模型的强大工具，它提供了直观且理论上合理的解释。")
print("通过本示例，您已经了解了SHAP的基本原理、各种可视化方法以及在实际机器学习流程中的应用。")
print("更多信息请查阅官方文档: https://shap.readthedocs.io/")
print("\n祝您使用愉快！")
print("=" * 80) 