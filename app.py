# flask文件路由配置
import numpy as np
from flask import Flask, request, jsonify
from flask import render_template
import pymysql
from flask_sqlalchemy import SQLAlchemy

from fangjia_xianxing import xianxing
from fangjia_juece import juece
from fangjia_shenjing import shenjing
from fangjia_svm import svm
from fangjia_jicheng import jicheng

app = Flask(__name__)

# 数据库链接变量
conn = pymysql.connect(
    host='127.0.0.1',
    user='root',
    passwd='Sx040716!',
    db='fz_ershoufang',
    charset='utf8'
)

cursor = conn.cursor()

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Sx040716!@localhost:3306/fz_ershoufang'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)


class Fangwu(db.Model):
    __tablename__ = 'fangwu'
    id = db.Column(db.INTEGER, autoincrement=True, primary_key=True)
    title = db.Column(db.String(200), unique=True, nullable=False)
    price_count = db.Column(db.String(80), unique=True, nullable=False)
    price = db.Column(db.String(20), unique=True, nullable=False)
    xiaoqu = db.Column(db.String(200), unique=True, nullable=False)
    quyu = db.Column(db.String(200), unique=True, nullable=False)
    home_num = db.Column(db.String(20), unique=True, nullable=False)
    size = db.Column(db.String(20), unique=True, nullable=False)
    chaoxiang = db.Column(db.String(20), unique=True, nullable=False)
    jianzao = db.Column(db.String(20), unique=True, nullable=False)
    tags = db.Column(db.String(20), unique=True, nullable=False)

    def __repr__(self):
        return '<Fangwu %r>' % self.name


# title, price_count, price[0], xiaoqu, quyu, home_num, size,chaoxiang, jianzao,tags


# 首页
@app.route("/")
def index1():
    return render_template("login.html")


@app.route('/dologin', methods=['GET', 'POST'])
def dologin():
    username = request.form.get('username')
    password = request.form.get('password')
    select_sql = 'select password from user where username = %s'
    cursor.execute(select_sql, (username))
    select_password = cursor.fetchone()
    print(select_password)
    if select_password is None or password != select_password[0]:
        return 'error'
    else:
        return 'success'


@app.route('/resgister', methods=['GET', 'POST'])
def resgister():
    print('12312312312')
    username = request.form.get('username')
    password = request.form.get('password')
    select_sql = "select username from user where username = %s"
    cursor.execute(select_sql, username)
    select_username = cursor.fetchone()
    if select_username is None:
        insert_sql = "insert into user(username,password) values (%s,%s)"
        cursor.execute(insert_sql, (username, password))
        conn.commit()
        return "注册成功"
    else:
        return "用户已存在"


@app.route("/index")
def index():
    sql = 'select count(*) from fangwu'
    cursor = conn.cursor()
    cursor.execute(sql)
    num = cursor.fetchall()
    cursor.close()
    num_chart = 2
    num_chart2 = 6
    return render_template("index.html", num=num[0][0], num_chart=num_chart, num_chart2=num_chart2)


#
@app.route("/jobCity")
def jobCity():
    sql = 'select jianzao,count(jianzao) from fangwu group by jianzao ORDER BY jianzao'
    cursor = conn.cursor()
    cursor.execute(sql)
    list = cursor.fetchall()
    sj_list = []
    zj_list = []
    for item in list:
        sj_list.append(item[0])
        zj_list.append(item[1])
    print(sj_list)
    return render_template("jobCity.html", sj_list=sj_list, zj_list=zj_list, list=list)


@app.route("/world")
def world():
    sql = 'select quyu,count(quyu),avg(price) from fangwu GROUP BY quyu'
    cursor = conn.cursor()
    cursor.execute(sql)
    list = cursor.fetchall()
    qy_list = []
    sl_list = []
    mean = []
    for item in list:
        qy_list.append(item[0])
        sl_list.append(item[1])
        mean.append(item[2])
    return render_template("world.html", mean=mean, qy_list=qy_list, sl_list=sl_list, list=list)


# 房价随时间变化
@app.route("/price")
def price():
    sql = 'select jianzao,avg(price) from fangwu group by jianzao ORDER BY jianzao'
    cursor = conn.cursor()
    cursor.execute(sql)
    list = cursor.fetchall()
    sj_list = []
    mean = []
    for item in list:
        sj_list.append(item[0])
        mean.append(int(item[1]))
    print(mean)
    return render_template("price.html", mean=mean, x_list=sj_list)


# 数量分析
@app.route("/num")
def num():
    sql = 'select home_num,count(home_num) from fangwu GROUP BY home_num ORDER BY home_num'
    cursor = conn.cursor()
    cursor.execute(sql)
    list = cursor.fetchall()
    result = []
    for item in list:
        result.append({'value': item[1], 'name': item[0]})

    return render_template("num.html", result=result)


# 词云
@app.route("/worldCloud")
def worldCloud():
    return render_template("worldCloud.html")


# 数据列表
@app.route("/table/<int:page_num>",methods=['GET','POST'])
def table(page_num=1):
    if request.method == 'POST':
        # 用户进行了搜索
        keyword = request.form['keyword']
        print(keyword)
        job_list = Fangwu.query.filter(Fangwu.title.like("%"+keyword+"%")).paginate(per_page=20, page=page_num, error_out=True)
        return render_template("table.html", job_list=job_list)
        # ...
    else:
        # 用户初次进入页面渲染
        job_list = Fangwu.query.paginate(per_page=20, page=page_num, error_out=True)
        return render_template("table.html", job_list=job_list)


# 房价预测
@app.route("/predicted")
def predicted():
    return render_template("predicted.html")

# 聚类
@app.route("/group")
def group():
    return render_template("group.html")


# 预测
@app.route("/Line")
def Line():
    return render_template("Line.html")


@app.route('/get_xianxing_predictions', methods=['GET'])
def get_xianxing_predictions():
    mse, rmse, predicted_price = xianxing()

    # 返回预测结果给前端
    return jsonify({
        'mse': mse,
        'rmse': rmse,
        'predicted_price': predicted_price
    })



@app.route('/get_juece_predictions', methods=['GET'])
def get_juece_predictions():
    mse, rmse, predicted_price = juece()

    # 返回预测结果给前端
    return jsonify({
        'mse': mse,
        'rmse': rmse,
        'predicted_price': predicted_price
    })

@app.route('/get_svm_predictions', methods=['GET'])
def get_svm_predictions():
    mse, rmse, predicted_price = svm()

    # 返回预测结果给前端
    return jsonify({
        'mse': mse,
        'rmse': rmse,
        'predicted_price': predicted_price
    })

@app.route('/get_shenjing_predictions', methods=['GET'])
def get_shenjing_predictions():
    mse, rmse, predicted_price = shenjing()

    # 返回预测结果给前端
    return jsonify({
        'mse': mse,
        'rmse': rmse,
        'predicted_price': predicted_price
    })

@app.route('/get_jicheng_predictions', methods=['GET'])
def get_jicheng_predictions():
    mse, rmse, predicted_price = jicheng()

    # 返回预测结果给前端
    return jsonify({
        'mse': mse,
        'rmse': rmse,
        'predicted_price': predicted_price
    })


if __name__ == '__main__':
    app.run(debug=True)

