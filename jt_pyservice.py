# -*- coding: utf-8 -*-
# @Author  : zhangcy
# @Email   : 604641446@qq.com
# @Time    : 2020/5/29 16:29
from gevent import monkey
monkey.patch_all()
import json
from flask import Flask,request,Response
import jt_algrith
from gevent import pywsgi

app=Flask(__name__)

@app.route("/jt_pyservice",methods=["GET","POST"])
def get_Aggl():
    return get_response(jt_algrith.getJson(request.get_json()))

def get_response(data):
    response = Response(
                response=json.dumps(data),
                status=200,
                mimetype='application/json'
            )
    return response
setting_f=open("jt_pyservice_setting.json",'r')
setting=json.load(setting_f)
setting_f.close()
print(setting)
server = pywsgi.WSGIServer((setting['ip'], int(setting['port'])), app)
server.serve_forever()