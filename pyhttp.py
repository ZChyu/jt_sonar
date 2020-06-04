# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:10:00 2018

@author: Zcy
"""

from urllib import request
import json
def getmodel(url,data):
    textmod = json.dumps(data).encode(encoding='utf-8')
#    textmod = parse.urlencode(textmod).encode(encoding='utf-8') #普通文本格式
    header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',"Content-Type": "application/json"}
    req = request.Request(url=url,data=textmod,headers=header_dict)
    res = request.urlopen(req)
    res = res.read()
    result=res
    return result