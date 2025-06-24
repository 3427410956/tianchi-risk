# -*- coding: utf-8 -*-
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt

print("=== 环境检查 ===")
print("Python版本:", sys.version)
print("当前工作目录:", os.getcwd())

try:
    # === 1. 路径设置 ===
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    print(f"数据目录: {DATA_DIR}")

    # === 2. 数据加载 ===
    train_path = os.path.join(DATA_DIR, 'train.csv')
    print(f"训练集路径: {train_path}")

    # 检查文件是否存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"文件不存在: {train_path}")

    # 读取数据（限制行数避免内存问题）
    train = pd.read_csv(train_path, nrows=100000)  # 先读10万行测试
    print(f"数据加载成功! 维度: {train.shape}")

    # === 3. 字段检查 ===
    required_columns = ['loanAmnt', 'interestRate', 'annualIncome', 'dti', 'revolBal', 'isDefault']
    missing_cols = [col for col in required_columns if col not in train.columns]

    if missing_cols:
        print("警告: 缺失字段 -", missing_cols)
        print("实际字段:", list(train.columns))
        # 尝试常见变体
        column_mapping = {
            'loanAmnt': ['loan_amnt', 'loanAmount', '贷款金额'],
            'isDefault': ['default', 'target', 'label']
        }
        for orig, alts in column_mapping.items():
            for alt in alts:
                if alt in train.columns:
                    train.rename(columns={alt: orig}, inplace=True)
                    print(f"已将字段 {alt} 重命名为 {orig}")

        # 再次检查
        missing_cols = [col for col in required_columns if col not in train.columns]
        if missing_cols:
            raise KeyError(f"关键字段缺失: {missing_cols}")

    # === 4. 数据处理 ===
    # 只保留需要的列
    features = ['loanAmnt', 'interestRate', 'annualIncome', 'dti', 'revolBal']
    target = 'isDefault'
    df = train[features + [target]].copy()

    # 处理缺失值
    print("缺失值统计:")
    print(df.isnull().sum())
    df = df.dropna()
    print(f"清洗后数据量: {len(df)}")

    # === 5. 模型训练 ===
    X = df[features]
    y = df[target]

    # 拆分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")

    # 使用LightGBM
    model = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # === 6. 模型评估 ===
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"\n=== 模型结果 ===")
    print(f"验证集AUC: {auc:.4f}")

    # ... [前面的代码保持不变] ...

    # === 7. 可视化 ===
    lgb.plot_importance(model, max_num_features=5, importance_type='gain')
    plt.title('特征重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'feature_importance.png'))
    print("特征重要性图已保存")

    # === 8. 测试集预测 ===
    test_path = os.path.join(DATA_DIR, 'testA.csv')
    if os.path.exists(test_path):
        test = pd.read_csv(test_path)
        print(f"测试集加载成功! 维度: {test.shape}")

        # 确保特征一致
        test_features = test[features]
        test_pred = model.predict_proba(test_features)[:, 1]

        submit = pd.DataFrame({
            'id': test['id'],
            'isDefault': test_pred
        })
        submit_path = os.path.join(BASE_DIR, 'tianchi_submit.csv')
        submit.to_csv(submit_path, index=False)
        print(f"提交文件已生成: {submit_path}")
    else:
        print("测试集不存在，跳过预测步骤")

except Exception as e:
    print("\n!!! 发生错误 !!!")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {str(e)}")
    print("\n=== 故障排除指南 ===")
    print("1. 检查文件路径是否正确")
    print("2. 确保数据文件存在")
    print("3. 确认字段名匹配")
    print("4. 检查依赖安装: pip install pandas lightgbm scikit-learn matplotlib")
    print("5. 减少数据量: 修改nrows参数")
finally:
    print("\n程序执行结束")
