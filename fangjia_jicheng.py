import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

def jicheng():
    # Step 1: 加载数据
    data = pd.read_csv('clean.csv')

    # 查看数据的前几行，确保加载正确
    # print(data.head())

    # Step 2: 数据预处理
    # 处理非数值数据列
    data['单价'] = pd.to_numeric(data['单价'], errors='coerce')
    data['房间数'] = pd.to_numeric(data['房间数'], errors='coerce')
    data['大小'] = pd.to_numeric(data['大小'], errors='coerce')
    data['建造时间'] = pd.to_numeric(data['建造时间'], errors='coerce')
    data['总价（万）'] = pd.to_numeric(data['总价（万）'], errors='coerce')

    # 将朝向列转换为数值类型
    data['朝向'] = data['朝向'].map({'南': 1, '北': 2, '南北': 3, '东': 4, '西': 5, '东南': 6, '西南': 7, '东北': 8, '西北': 9}).fillna(0)

    # 将建造时间列转换为年份
    data['建造时间'] = pd.to_datetime(data['建造时间'], errors='coerce').dt.year

    # Step 3: 只对数值型数据填充缺失值
    numeric_cols = data.select_dtypes(include=[np.number]).columns  # 获取所有数值型列
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Step 4: 选择特征列和目标列
    X = data[['单价', '房间数', '大小', '朝向', '建造时间']]
    y = data['总价（万）']

    # Step 5: 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 6: 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Step 7: 创建集成学习模型

    # 基础学习器：随机森林回归、梯度提升回归、XGBoost、LightGBM
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)

    # 元学习器：线性回归
    meta_model = LinearRegression()

    # Stack模型
    stacking_model = StackingRegressor(
        estimators=[('rf', rf_model), ('gb', gb_model), ('xgb', xgb_model), ('lgb', lgb_model)],
        final_estimator=meta_model
    )

    # Step 8: 训练模型
    stacking_model.fit(X_train, y_train)

    # Step 9: 进行预测
    y_pred = stacking_model.predict(X_test)

    # Step 10: 评估模型性能
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # print(f'Mean Squared Error: {mse}')
    # print(f'Root Mean Squared Error: {rmse}')

    # 预测样例
    sample_data = X_test[0:1]
    predicted_price = stacking_model.predict(sample_data)
    # print(f'Predicted price for the sample: {predicted_price[0]} 万')

    # 绘制散点图
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices (in million)')
    plt.ylabel('Predicted Prices (in million)')
    plt.title('Actual vs Predicted House Prices')

    # 绘制拟合直线
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # 保存图片
    plt.savefig('static/assets/img/modelView/集成学习.jpg')

    plt.show()



    # 将结果传递给前端模板
    return (mse, rmse,predicted_price[0])

if __name__ == '__main__':
    print(jicheng())


