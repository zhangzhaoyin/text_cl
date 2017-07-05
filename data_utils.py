# -*- coding: utf-8 -*-
# @Author: Zhaoyin Zhang
# @Date:2017/7/4 14:40 
# @Contact: 940942500@qq.com

import jieba
import re
import json


def segWord(x):
    seglist = " ".join(jieba.cut(x))
    return seglist


def clearChar(x):
    x = re.sub("（.*?）|【.*?】|\(.*?\)", "", str(x))
    charList = re.sub("[^\u4e00-\u9fa5]", "", x)
    return charList


def cut(x):
    return segWord(clearChar(x))


def getLabels(industry=None):
    industry = '{"data":[{"primInduCode":"D","primInduName":"电力、热力、燃气及水生产和供应业",' \
               '"secList":[{"secnduName":"燃气生产和供应业","secInduCode":"45"},{"secnduName":"电力、热力生产和供应业","secInduCode":"44"},' \
               '{"secnduName":"水的生产和供应业","secInduCode":"46"}]},{"primInduCode":"E","primInduName":"建筑业",' \
               '"secList":[{"secnduName":"建筑安装业","secInduCode":"49"},{"secnduName":"土木工程建筑业","secInduCode":"48"},' \
               '{"secnduName":"房屋建筑业","secInduCode":"47"},{"secnduName":"建筑装饰和其他建筑业","secInduCode":"50"}]},' \
               '{"primInduCode":"F","primInduName":"批发和零售业","secList":[{"secnduName":"批发业","secInduCode":"51"},' \
               '{"secnduName":"零售业","secInduCode":"52"}]},{"primInduCode":"G","primInduName":"交通运输、仓储和邮政业",' \
               '"secList":[{"secnduName":"仓储业","secInduCode":"59"},{"secnduName":"装卸搬运和运输代理业","secInduCode":"58"},' \
               '{"secnduName":"管道运输业","secInduCode":"57"},{"secnduName":"航空运输业","secInduCode":"56"},{"secnduName":"水上运输业",' \
               '"secInduCode":"55"},{"secnduName":"邮政业","secInduCode":"60"},{"secnduName":"铁路运输业","secInduCode":"53"},' \
               '{"secnduName":"道路运输业","secInduCode":"54"}]},{"primInduCode":"A","primInduName":"农、林、牧、渔业",' \
               '"secList":[{"secnduName":"渔业","secInduCode":"04"},{"secnduName":"农、林、牧、渔服务业","secInduCode":"05"},' \
               '{"secnduName":"农业","secInduCode":"01"},{"secnduName":"林业","secInduCode":"02"},{"secnduName":"畜牧业",' \
               '"secInduCode":"03"}]},{"primInduCode":"B","primInduName":"采矿业","secList":[{"secnduName":"开采辅助活动",' \
               '"secInduCode":"11"},{"secnduName":"其他采矿业","secInduCode":"12"},{"secnduName":"黑色金属矿采选业","secInduCode":"08"},' \
               '{"secnduName":"有色金属矿采选业","secInduCode":"09"},{"secnduName":"煤炭开采和洗选业","secInduCode":"06"},' \
               '{"secnduName":"石油和天然气开采业","secInduCode":"07"},{"secnduName":"非金属矿采选业","secInduCode":"10"}]},' \
               '{"primInduCode":"C","primInduName":"制造业","secList":[{"secnduName":"专用设备制造业","secInduCode":"35"},' \
               '{"secnduName":"汽车制造业","secInduCode":"36"},{"secnduName":"金属制品业","secInduCode":"33"},{"secnduName":"通用设备制造业",' \
               '"secInduCode":"34"},{"secnduName":"计算机、通信和其他电子设备制造业","secInduCode":"39"},{"secnduName":"铁路、船舶、航空航天和其他运输设备制造业",' \
               '"secInduCode":"37"},{"secnduName":"电气机械和器材制造业","secInduCode":"38"},{"secnduName":"金属制品、机械和设备修理业","secInduCode":"43"},' \
               '{"secnduName":"废弃资源综合利用业","secInduCode":"42"},{"secnduName":"其他制造业","secInduCode":"41"},{"secnduName":"仪器仪表制造业",' \
               '"secInduCode":"40"},{"secnduName":"造纸和纸制品业","secInduCode":"22"},{"secnduName":"印刷和记录媒介复制业","secInduCode":"23"},' \
               '{"secnduName":"文教、工美、体育和娱乐用品制造业","secInduCode":"24"},{"secnduName":"石油加工、炼焦和核燃料加工业","secInduCode":"25"},' \
               '{"secnduName":"化学原料和化学制品制造业","secInduCode":"26"},{"secnduName":"医药制造业","secInduCode":"27"},{"secnduName":"化学纤维制造业",' \
               '"secInduCode":"28"},{"secnduName":"橡胶和塑料制品业","secInduCode":"29"},{"secnduName":"非金属矿物制品业","secInduCode":"30"},' \
               '{"secnduName":"有色金属冶炼和压延加工业","secInduCode":"32"},{"secnduName":"黑色金属冶炼和压延加工业","secInduCode":"31"},' \
               '{"secnduName":"皮革、毛皮、羽毛及其制品和制鞋业","secInduCode":"19"},{"secnduName":"纺织业","secInduCode":"17"},' \
               '{"secnduName":"纺织服装、服饰业","secInduCode":"18"},{"secnduName":"酒、饮料和精制茶制造业","secInduCode":"15"},{"secnduName":"烟草制品业",' \
               '"secInduCode":"16"},{"secnduName":"农副食品加工业","secInduCode":"13"},{"secnduName":"食品制造业","secInduCode":"14"},' \
               '{"secnduName":"家具制造业","secInduCode":"21"},{"secnduName":"木材加工和木、竹、藤、棕、草制品业","secInduCode":"20"}]},' \
               '{"primInduCode":"L","primInduName":"租赁和商务服务业","secList":[{"secnduName":"租赁业","secInduCode":"71"},{"secnduName":"商务服务业",' \
               '"secInduCode":"72"}]},{"primInduCode":"M","primInduName":"科学研究和技术服务业","secList":[{"secnduName":"研究和试验发展","secInduCode":"73"},' \
               '{"secnduName":"专业技术服务业","secInduCode":"74"},{"secnduName":"科技推广和应用服务业","secInduCode":"75"}]},{"primInduCode":"N",' \
               '"primInduName":"水利、环境和公共设施管理业","secList":[{"secnduName":"公共设施管理业","secInduCode":"78"},{"secnduName":"生态保护和环境治理业",' \
               '"secInduCode":"77"},{"secnduName":"水利管理业","secInduCode":"76"}]},{"primInduCode":"O","primInduName":"居民服务、修理和其他服务业",' \
               '"secList":[{"secnduName":"居民服务业","secInduCode":"79"},{"secnduName":"机动车、电子产品和日用产品修理业","secInduCode":"80"},' \
               '{"secnduName":"其他服务业","secInduCode":"81"}]},{"primInduCode":"H","primInduName":"住宿和餐饮业","secList":[{"secnduName":"餐饮业",' \
               '"secInduCode":"62"},{"secnduName":"住宿业","secInduCode":"61"}]},{"primInduCode":"I","primInduName":"信息传输、软件和信息技术服务业",' \
               '"secList":[{"secnduName":"互联网和相关服务","secInduCode":"64"},{"secnduName":"软件和信息技术服务业","secInduCode":"65"},' \
               '{"secnduName":"电信、广播电视和卫星传输服务","secInduCode":"63"}]},{"primInduCode":"J","primInduName":"金融业",' \
               '"secList":[{"secnduName":"资本市场服务","secInduCode":"67"},{"secnduName":"货币金融服务","secInduCode":"66"},{"secnduName":"其他金融业",' \
               '"secInduCode":"69"},{"secnduName":"保险业","secInduCode":"68"}]},{"primInduCode":"K","primInduName":"房地产业",' \
               '"secList":[{"secnduName":"房地产业","secInduCode":"70"}]},{"primInduCode":"T","primInduName":"国际组织","secList":[{"secnduName":"国际组织",' \
               '"secInduCode":"96"}]},{"primInduCode":"Q","primInduName":"卫生和社会工作","secList":[{"secnduName":"卫生","secInduCode":"83"},' \
               '{"secnduName":"社会工作","secInduCode":"84"}]},{"primInduCode":"P","primInduName":"教育","secList":[{"secnduName":"教育","secInduCode":"82"}]},' \
               '{"primInduCode":"S","primInduName":"公共管理、社会保障和社会组织","secList":[{"secnduName":"基层群众自治组织","secInduCode":"95"},' \
               '{"secnduName":"群众团体、社会团体和其他成员组织","secInduCode":"94"},{"secnduName":"社会保障","secInduCode":"93"},{"secnduName":"人民政协、民主党派",' \
               '"secInduCode":"92"},{"secnduName":"国家机构","secInduCode":"91"},{"secnduName":"中国共产党机关","secInduCode":"90"}]},{"primInduCode":"R",' \
               '"primInduName":"文化、体育和娱乐业","secList":[{"secnduName":"体育","secInduCode":"88"},{"secnduName":"娱乐业","secInduCode":"89"},' \
               '{"secnduName":"广播、电视、电影和影视录音制作业","secInduCode":"86"},{"secnduName":"文化艺术业","secInduCode":"87"},{"secnduName":"新闻和出版业","secInduCode":"85"}]}]}'

    industrys = []
    for i in json.loads(industry)["data"]:
        for j in i["secList"]:
            industrys.append(j)
    return industrys



