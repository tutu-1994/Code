import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def svm():

    # Step 1: 加载数据
    data = pd.read_csv('clean.csv')

    # 查看数据的前几行，确保加载正确
    # print(data.head())

    # Step 2: 数据预处理
    # 检查并处理数据类型
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

    # Step 5: 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: 特征缩放 - SVR对特征的缩放较为敏感
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 7: 训练支持向量回归模型
    model = SVR(kernel='rbf')  # 使用径向基函数（RBF）内核
    model.fit(X_train_scaled, y_train)

    # Step 8: 进行预测
    y_pred = model.predict(X_test_scaled)

    # Step 9: 评估模型性能
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # print(f'Mean Squared Error: {mse}')
    # print(f'Root Mean Squared Error: {rmse}')

    # 预测样例
    sample_data = X_test.iloc[0:1]
    predicted_price = model.predict(scaler.transform(sample_data))
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
    plt.savefig('static/assets/img/modelView/支持向量机SVM.jpg')

    plt.show()

    # 将结果传递给前端模板
    return (mse, rmse, predicted_price[0])