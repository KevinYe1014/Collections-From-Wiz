````python
# coding: utf-8
# coding: utf-8
"""Train tickets query via command-line.

Usage:
    Tickets [-gdtkz] <from> <to> <date>



Options:
    -h,--help   显示帮助菜单
    -g          高铁
    -d          动车
    -t          特快
    -k          快速
    -z          直达

Example:
    tickets beijing shanghai 2016-08-25
"""



import requests
from docopt import docopt
from prettytable import PrettyTable
from time import sleep
import re
import time,datetime


def get_station_code():
    url = 'https://kyfw.12306.cn/otn/resources/js/framework/station_name.js?station_version=1.9098'
    requests.packages.urllib3.disable_warnings()
    text = requests.get(url ,verify=False).text
    inf = text[:-2].split('@')[1:]

    stations = {}
    stations_research = {}
    for record in inf:
        rlist = record.split("|")
        stations[int(rlist[-1])] = {"name": rlist[1], "search": rlist[2], "fullname": rlist[3], "firstname": rlist[4]}
        stations_research[rlist[1]] = rlist[2]
    return stations_research



def Price_Checi_Type(checi,checitype,pricedata):
    if checi=='' or checi=='-':
        return '-'
    elif checitype not in pricedata.keys():
        return checi
    elif '/' in checi:
        return str(checi.split('/')[0])
    else:
        return '{0}/{1}'.format(checi,pricedata[checitype])



def decode(rows,stations_research,arguments):
    result=[]
    checi_type=[]
    '''
    -g          高铁
    -d          动车
    -t          特快
    -k          快速
    -z          直达
    '''
    for char in 'gdtkz':
        _char='-{}'.format(char)
        if arguments[_char]:
            checi_type.append(char.upper())

    for i in rows:
        list = i.split("|")
        checi = list[3]
        chufa = [k for k,v in stations_research.items() if v==list[6]][0]
        mudi = [k for k,v in stations_research.items() if v==list[7]][0]
        ftime = list[8]
        dtime = list[9]
        time=list[10]

        sw = list[32] if list[25]=='' else list[25]
        yd = list[31]
        ed=list[30]

        rw = list[23]
        dw=list[33]
        yw = list[28]

        # rz=list[27]
        yz = list[29]
        wuzuo = list[26]


        if checi[0] in checi_type or len(checi_type)==0:

            # region 车票信息
            train_no = list[2]
            from_station_no = list[16]
            to_station_no = list[17]
            seat_types = list[35]
            train_date = '{0}-{1}-{2}'.format(list[13][:4], list[13][4:6], list[13][6:])
            url = 'https://kyfw.12306.cn/otn/leftTicket/queryTicketPrice?train_no={0}&from_station_no={1}&to_station_no={2}&seat_types={3}&train_date={4}'. \
                format(train_no, from_station_no, to_station_no, seat_types, train_date)
            sleep(0.1)

            _pricedata = requests.get(url)
            _pricedata.encoding = 'utf-8'
            pricedata = _pricedata.json()['data']

            _sw=Price_Checi_Type(sw,'A9',pricedata)
            _sw=Price_Checi_Type(_sw,'P',pricedata)
            _yd=Price_Checi_Type(yd,'M',pricedata)
            _ed=Price_Checi_Type(ed,'O',pricedata)

            _rw=Price_Checi_Type(rw,'A4',pricedata)
            _dw=Price_Checi_Type(dw,'F',pricedata)
            _yw=Price_Checi_Type(yw,'A3',pricedata)

            _yz=Price_Checi_Type(yz,'A1',pricedata)
            _wuzuo=Price_Checi_Type(wuzuo,'WZ',pricedata)

            result.append((checi, chufa, mudi, ftime, dtime,time, _sw, _yd, _ed,_rw,_dw,_yw, _yz, _wuzuo))
    return result



def get_color(_str,color_num):
    return "\033[{};".format(str(color_num))+"0m"+_str+"\033[0m"


def pretty_table(results):
    table=PrettyTable(["车次","出发站","目的站","发车时间","到达时间","历时","商务座","一等座","二等座","软卧","动卧","硬卧","硬座","无座"])
    color_num=31
    train_count=1
    for i in results:
        table.add_row([get_color(i[0],color_num),get_color(i[1],color_num),get_color(i[2],color_num),get_color(i[3],color_num),get_color(i[4],color_num),
                       get_color(i[5], color_num),get_color(i[6],color_num),get_color(i[7],color_num),get_color(i[8],color_num),get_color(i[9],color_num),
                       get_color(i[10], color_num),get_color(i[11],color_num),get_color(i[12],color_num),get_color(i[13],color_num)])
        color_num+=1
        train_count+=1
        if color_num>36:
            color_num=31
    print(table)
    print("共查询到列车次数：{}".format(str(train_count-1)))



def _Train_Search():
    """command-line interface"""
    arguments = docopt(__doc__)

    stations_research=get_station_code()


    from_station=stations_research.get(arguments['<from>'])
    to_station=stations_research.get(arguments['<to>'])
    date = arguments['<date>']
    nowtime=time.strftime('%Y-%m-%d',time.localtime(time.time()))
    nowtime=datetime.datetime.strptime(nowtime, "%Y-%m-%d")
    if from_station==None:
        print("请确认出发站：{}是否输入正确...".format(arguments['<from>']))
    elif to_station==None:
        print("请确认目的站：{}是否输入正确...".format(arguments['<to>']))
    elif len(re.findall('\d{4}-\d{2}-\d{2}',date))==0 :
        print("请确认搜索日期：{}是否输入正确...".format(arguments['<date>']))
    elif (datetime.datetime.strptime(date, "%Y-%m-%d")-nowtime).days <0:
        print("请确认搜索日期是否小于今日日期...")
    elif (datetime.datetime.strptime(date, "%Y-%m-%d")-nowtime).days >29:
        print("只能查询今日及后面30日以内...")
    else:
        url = 'https://kyfw.12306.cn/otn/leftTicket/query?leftTicketDTO.train_date={0}&leftTicketDTO.from_station={1}&leftTicketDTO.to_station={2}&purpose_codes=ADULT'.format(
            date, from_station, to_station)
        rows = requests.get(url, verify=False).json()['data']['result']
        if len(rows) == 0:
            print("没有从{0}到{1}的列车，请考虑换乘线路...".format(arguments['<from>'], arguments['<to>']))
        else:
            results = decode(rows, stations_research, arguments)
            if len(results) == 0:
                print("从{0}到{1}没有你需要查询的列车类型，请选择其他类型列车...".format(arguments['<from>'], arguments['<to>']))
            else:
                pretty_table(results)

if __name__=='__main__':
    _Train_Search()


    
    
安庆 北京 2019-05-09
````

