import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna
import shap
import joblib
import os

# 设置随机种子确保结果可重复
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    """加载数据并进行预处理"""
    # 加载沪深300指数数据
    hs300_index = pd.read_csv("hs300_index.csv", parse_dates=['trade_date'], index_col=0)
    
    # 加载趋势标签
    l_span = np.arange(10, 40)
    import MachineLearningforAssetManagers as mlam
    hs300_trend = mlam.getTrendLabel(hs300_index.close, l_span=l_span)
    
    # 合并数据
    data = hs300_index.copy()
    data['t_value'] = hs300_trend['t_value']
    data = data.dropna() # 去除空值
    
    # 创建特征
    data = create_features(data)
    
    return data

def create_features(data):
    """创建特征"""
    df = data.copy()
    
    # 价格特征
    df['close_to_open'] = df['close'] / df['open'] - 1
    df['high_to_low'] = df['high'] / df['low'] - 1
    df['close_to_high'] = df['close'] / df['high'] - 1
    df['close_to_low'] = df['close'] / df['low'] - 1
    
    # 移动平均线
    for window in [5, 10, 20, 30, 60]:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'close_to_ma_{window}'] = df['close'] / df[f'ma_{window}'] - 1
        
        # 波动率特征
        df[f'volatility_{window}'] = df['close'].rolling(window=window).std() / df['close']
    
    # RSI指标
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    for period in [6, 14, 21]:
        df[f'rsi_{period}'] = calculate_rsi(df['close'], period)
    
    # 动量指标
    for window in [5, 10, 20]:
        df[f'momentum_{window}'] = df['close'].pct_change(periods=window)
    
    # 成交量特征
    df['volume_change'] = df['vol'].pct_change()
    for window in [5, 10, 20]:
        df[f'volume_ma_{window}'] = df['vol'].rolling(window=window).mean()
        df[f'volume_to_ma_{window}'] = df['vol'] / df[f'volume_ma_{window}'] - 1
    
    # 去除NaN值
    df = df.dropna()
    
    return df

def add_historical_features(data, features, lookback_periods=[1, 2, 3, 5, 10]):
    """
    添加历史特征数据，类似LSTM的输入
    
    参数:
    data: 包含原始特征的DataFrame
    features: 需要添加历史数据的特征列表
    lookback_periods: 回看的时间窗口列表
    
    返回:
    包含历史特征的DataFrame
    """
    df = data.copy()
    
    # 确保数据按时间排序
    df = df.sort_index()
    
    # 为每个特征添加历史数据
    for feature in features:
        for period in lookback_periods:
            # 添加t-1, t-2, ... , t-n的历史数据
            df[f'{feature}_lag_{period}'] = df[feature].shift(period)
    
    # 去除NaN值
    df = df.dropna()
    
    return df

def prepare_train_test_data(data, test_size=0.2, target_col='t_value'):
    """准备训练集和测试集"""
    # 按时间排序
    data = data.sort_index()
    
    # 确定特征和目标变量
    y = data[target_col]
    
    # 移除不需要的列
    X = data.drop([target_col, 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'], axis=1, errors='ignore')
    
    # 时间序列分割
    split_idx = int(len(data) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test

def create_dmatrix_datasets(X_train, y_train, X_test, y_test):
    """
    创建XGBoost的DMatrix数据集，用于加速训练
    
    参数:
    X_train, y_train: 训练集特征和标签
    X_test, y_test: 测试集特征和标签
    
    返回:
    DMatrix格式的训练集和测试集
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    return dtrain, dtest

def objective_dmatrix(trial, dtrain, dvalid):
    """Optuna目标函数，使用DMatrix进行超参数优化"""
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'seed': RANDOM_SEED
    }
    
    if param['booster'] in ['gbtree', 'dart']:
        param['max_depth'] = trial.suggest_int('max_depth', 3, 12)
        param['eta'] = trial.suggest_float('eta', 0.01, 0.3, log=True)
        param['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
        param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        param['tree_method'] = 'hist'  # 使用直方图算法加速
    
    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.5)
        param['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)
    
    # 使用early stopping避免过拟合
    num_round = trial.suggest_int('num_round', 50, 500)
    
    # 训练模型
    evals_result = {}
    bst = xgb.train(
        param, 
        dtrain, 
        num_round, 
        evals=[(dtrain, 'train'), (dvalid, 'validation')],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False
    )
    
    # 返回验证集上的最佳RMSE
    return min(evals_result['validation']['rmse'])

def optimize_hyperparameters_dmatrix(X_train, y_train, n_trials=100):
    """使用DMatrix和Optuna优化XGBoost模型的超参数"""
    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = []
    best_params = None
    best_score = float('inf')
    best_num_round = 0
    
    for train_idx, val_idx in tscv.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 创建DMatrix
        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
        dvalid = xgb.DMatrix(X_val_cv, label=y_val_cv)
        
        # 创建Optuna研究
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective_dmatrix(trial, dtrain, dvalid), n_trials=n_trials)
        
        # 记录结果
        cv_scores.append(study.best_value)
        if study.best_value < best_score:
            best_score = study.best_value
            best_params = study.best_params
            if 'num_round' in study.best_params:
                best_num_round = study.best_params['num_round']
                # 从参数中移除num_round，因为它是xgb.train的参数，不是模型参数
                best_params.pop('num_round')
    
    print(f"交叉验证RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"最佳参数: {best_params}")
    print(f"最佳迭代次数: {best_num_round}")
    
    return best_params, best_num_round

def train_model_dmatrix(dtrain, dtest, params=None, num_round=100):
    """使用DMatrix训练XGBoost模型"""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'seed': RANDOM_SEED
        }
    
    # 训练模型
    evals_result = {}
    model = xgb.train(
        params, 
        dtrain, 
        num_round, 
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=True
    )
    
    return model, evals_result

def evaluate_model_dmatrix(model, dtest, y_test):
    """评估使用DMatrix训练的模型性能"""
    y_pred = model.predict(dtest)
    
    # 获取索引
    test_index = y_test.index
    
    # 计算各种指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"测试集评估结果:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(test_index, y_test.values, label='实际值')
    plt.plot(test_index, y_pred, label='预测值')
    plt.title('沪深300趋势预测结果')
    plt.xlabel('日期')
    plt.ylabel('t_value')
    plt.legend()
    plt.grid(True)
    plt.savefig('hs300_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制训练曲线（如果有evals_result）
    if hasattr(model, 'evals_result'):
        evals_result = model.evals_result()
        if evals_result:
            epochs = len(evals_result['train']['rmse'])
            x_axis = range(0, epochs)
            plt.figure(figsize=(10, 6))
            plt.plot(x_axis, evals_result['train']['rmse'], label='训练集')
            plt.plot(x_axis, evals_result['test']['rmse'], label='测试集')
            plt.legend()
            plt.xlabel('迭代次数')
            plt.ylabel('RMSE')
            plt.title('训练过程中的RMSE变化')
            plt.grid()
            plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred
    }

def analyze_feature_importance_dmatrix(model, feature_names):
    """分析DMatrix模型的特征重要性"""
    # 获取特征重要性
    importance = model.get_score(importance_type='weight')
    
    # 将特征重要性转换为DataFrame，确保所有特征都有值
    all_features = {feature: importance.get(feature, 0) for feature in feature_names}
    importance_df = pd.DataFrame({'feature': list(all_features.keys()), 'importance': list(all_features.values())})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    plt.title('特征重要性')
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印特征重要性
    print("\n特征重要性排名:")
    for i, (feature, imp) in enumerate(zip(importance_df['feature'], importance_df['importance'])):
        print(f"{i+1}. {feature}: {imp:.4f}")
    
    return importance_df

def shap_analysis_dmatrix(model, X_test):
    """使用SHAP进行DMatrix模型解释"""
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    # 将X_test转换为numpy数组，用于SHAP分析
    X_test_values = X_test.values
    feature_names = X_test.columns
    
    # 对测试集计算SHAP值
    shap_values = explainer.shap_values(X_test_values)
    
    # 可视化摘要图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_values, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP依赖图 - 选择最重要的特征
    importance = model.get_score(importance_type='weight')
    top_features_idx = np.argsort([-importance.get(f, 0) for f in feature_names])[:3]
    top_features = [feature_names[i] for i in top_features_idx if i < len(feature_names)]
    
    for i, feature in enumerate(top_features):
        if i >= len(feature_names):
            continue
        feature_idx = list(feature_names).index(feature)
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, shap_values, X_test_values, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 保存单个样本的SHAP力图
    plt.figure(figsize=(16, 6))
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_test_values[0,:], feature_names=feature_names, matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return shap_values

def save_model_dmatrix(model, model_path='hs300_xgb_model.json'):
    """保存DMatrix模型到文件"""
    model.save_model(model_path)
    print(f"模型已保存到 {model_path}")

def main():
    """主函数"""
    # 加载数据
    print("正在加载数据...")
    data = load_data()
    print(f"数据加载完成，共有 {len(data)} 条记录")
    
    # 添加历史特征
    print("正在添加历史特征...")
    # 选择需要添加历史数据的特征
    historical_features = [
        'close_to_open', 'high_to_low', 'close_to_high', 'close_to_low',
        'rsi_14', 'momentum_10', 'volume_change'
    ]
    data_with_history = add_historical_features(data, historical_features, lookback_periods=[1, 3, 5, 10])
    print(f"添加历史特征后，共有 {len(data_with_history)} 条记录，{data_with_history.shape[1]} 个特征")
    
    # 准备训练集和测试集
    print("正在准备训练集和测试集...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(data_with_history)
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 创建DMatrix数据集
    print("正在创建DMatrix数据集...")
    dtrain, dtest = create_dmatrix_datasets(X_train, y_train, X_test, y_test)
    
    # 如果需要进行参数优化（可能耗时）
    use_optimized_params = True
    if use_optimized_params:
        print("正在使用DMatrix优化超参数...")
        try:
            best_params, best_num_round = optimize_hyperparameters_dmatrix(X_train, y_train, n_trials=50)
        except Exception as e:
            print(f"参数优化出错: {e}")
            print("使用默认参数继续...")
            best_params = None
            best_num_round = 100
    else:
        # 使用预设参数
        best_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'seed': RANDOM_SEED
        }
        best_num_round = 100
        print("使用预设参数...")
    
    # 训练模型
    print("正在使用DMatrix训练模型...")
    model, evals_result = train_model_dmatrix(dtrain, dtest, best_params, best_num_round)
    
    # 评估模型
    print("正在评估模型...")
    results = evaluate_model_dmatrix(model, dtest, y_test)
    
    # 分析特征重要性
    print("正在分析特征重要性...")
    analyze_feature_importance_dmatrix(model, X_train.columns)
    
    # SHAP分析
    print("正在进行SHAP分析...")
    try:
        shap_values = shap_analysis_dmatrix(model, X_test)
    except Exception as e:
        print(f"SHAP分析出错: {e}")
    
    # 保存模型
    save_model_dmatrix(model)
    
    print("所有分析完成！")
    return model, results

if __name__ == "__main__":
    main() 