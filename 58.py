import random

import requests
from lxml import html
import time
import re

import csv
etree = html.etree
def getip():  # 随机ip
    with open("ipdaili.txt") as f:
        iplist = f.readlines()
    proxy = iplist[random.randint(0, len(iplist) - 1)]
    proxy = proxy.replace("\n", "")
    proxies = {
        'http': 'http://' + str(proxy)
    }
    return proxies
fp = open('福州sz.csv', 'a+', encoding='utf-8-sig',newline='')
csv_writer = csv.writer(fp, dialect='excel')
csv_writer.writerow(["标题", "总价（万）", "单价", "小区", "区域", "房间数", "大小",  "朝向","建造时间","标签"])

for i in range(13,50):
    time.sleep(random.randint(3,7))
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
                      '87.0.4280.141 Safari/537.36',
        'cookie':'f=n; commontopbar_new_city_info=4%7C%E6%B7%B1%E5%9C%B3%7Csz; f=n; commontopbar_new_city_info=4%7C%E6%B7%B1%E5%9C%B3%7Csz; commontopbar_ipcity=szkunshan%7C%E6%98%86%E5%B1%B1%7C0; userid360_xml=39566F53F971BCBA07BC89E58C47977F; time_create=1682516936045; id58=CocH3mLuaj8T7inIDhOyAg==; 58tj_uuid=bb3c2090-1229-4602-8409-5dc74d30a567; als=0; wmda_uuid=e9f4ac67384451bfa4a54795c26dbf0e; wmda_new_uuid=1; aQQ_ajkguid=68A39522-6AAE-4398-90AC-932D56B12DF4; sessid=EEA65CFA-490F-4CCC-B130-708C91DCE63F; ajk-appVersion=; xxzl_smartid=4abd17decef21cd5365b86f6440b9959; __bid_n=1851edaeab74c50b334207; FEID=v10-0e08576510d8725a200b083b8f16beb601297148; __xaf_fpstarttimer__=1671368587784; __xaf_thstime__=1671368587849; __xaf_fptokentimer__=1671368587896; __utma=253535702.1421825006.1676019160.1676019160.1676019160.1; __utmz=253535702.1676019160.1.1.utmcsr=szkunshan.58.com|utmccn=(referral)|utmcmd=referral|utmcct=/; FPTOKEN=c6kmeFYZ1KBe0kRCm9eh8DSlE43JVDydjB82Mgjq0Oxxk6Fvx14dkn0o/KW7RH4rYUyXK2Kwu6EJpUQ4BOC4uyL7LeKqDl7w7NgVKR7O251uU1Ec8Qrm3T69aIqFt5O8+lnJloG/5mlQap0Ctac/3fkp6bn1NiP41Wu9bDeZLjM2rrcZgjGixzgXj3DHIulSFKw0+QAghtZJrZ531w1AGeq/xDMDmnD+AW/uo/BXwxvLA3ACXiXbYeUpkyTpnZDVwdENw9jzvc4U/HjaEO1rNd3YkiQ7t93/ch2vtnYENYkwhCECfsV/77ezoemRX/geW29qfqW4SOGU8sj0Y6y9Z5QJx3cNtKjNge6Mg6txbmSlYRxzlies7VEgKPLnIGbukNvc9obd+RIXxod0079qRQ==|PsMZdWpf50qEcN63yAwakgsW/v56Yiyhvmqcv69j4Bg=|10|7f3b63cde53de3ea880815006f7e2166; wmda_visited_projects=%3B1731916484865%3B11187958619315%3B10104579731767%3B1409632296065; ppStore_fingerprint=62C842D9E4675F642851410B8F2B150BF8BDAD7783DA3FA5%EF%BC%BF1676020485486; ctid=4; fzq_h=63236f9b2d64efc2715cd3ea19aa75ff_1679924933304_8ecd46b668d94c6c9d69e03a22808461_1879081897; new_uv=10; utm_source=; spm=; init_refer=; 58home=szkunshan; new_session=0; xxzl_cid=652550fcb171415b878dd2e96485b83c; xxzl_deviceid=50gTBxPHMvdyI1AiXAhETghpvcSd4/Zo6E/9OJwqLl6480aSNJIeVU2nHKFlDMrY'
    }
    url ='https://fz.58.com/ershoufang/p{}?'.format(i)
    page_text = requests.get(url=url,headers=headers, proxies=getip()).text

    tree = etree.HTML(page_text)
    li_list = tree.xpath('//section[@class="list"][1]/div')

    for li in li_list:
        title = li.xpath('normalize-space(./a/div[2]/div/div/h3/text())')
        price_count = li.xpath('normalize-space(./a/div[2]/div[2]/p[1]/span[1]/text())')
        price = li.xpath('./a/div[2]/div[2]/p[2]/text()')
        xiaoqu = li.xpath('normalize-space(./a/div[2]/div/section/div[2]/p/text())')
        quyu = li.xpath('normalize-space(./a/div[2]/div/section/div[2]/p[2]/span[1]/text())')
        home_num = li.xpath('normalize-space(./a/div[2]/div/section/div/p/span[1]/text())')
        size = li.xpath('normalize-space(./a/div[2]/div/section/div/p[2]/text())').replace(' ','')
        chaoxiang = li.xpath('normalize-space(./a/div[2]/div/section/div/p[3]/text())')
        jianzao = li.xpath('normalize-space(./a/div[2]/div[1]/section/div[1]/p[5]/text())')
        tags = li.xpath('./a/div[2]/div[1]/section/div[3]//text()')
        csv_writer.writerow([title, price_count, price[0] if price else 0, xiaoqu, quyu, home_num, size,chaoxiang, jianzao,tags])
    print(f'第{i}页爬取')
        # filename.close()
fp.close()
print("over")


