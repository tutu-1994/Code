#将csv文件转存到mysql中

import pymysql
import pandas as pd
from sqlalchemy import create_engine
import csv

def csvtomusql():
    df = pd.read_csv("clean.csv")
    df.columns=['title', 'price_count', 'price', 'xiaoqu', 'quyu', 'home_num', 'size','chaoxiang','jianzao','tags']
    engine = create_engine('mysql+pymysql://root:123456@localhost/fz_ershoufang?charset=utf8')
    df.to_sql('fangwu',engine,if_exists='replace',index=False)

if __name__ == '__main__':
    csvtomusql()