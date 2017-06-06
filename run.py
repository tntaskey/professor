#!/bin/python
import bottle as bottle
from bottle import *
import urlparse # if we're pre-2.6, this will not include parse_qs
try:
            from urlparse import parse_qs
except ImportError: # old version, grab it from cgi
            from cgi import parse_qs
            urlparse.parse_qs = parse_qs
#Static
@route('/<filename:path>')
def send_static(filename):
    return static_file(filename, root='static/')

#Template
@route('/')
def main():
    return template('index.tpl')

@post("/GID")
def post_gid():
    USER_IN = request.query.get("gid") or ""
    print("x")
    return template('accounts.tpl', USER_IN=USER_IN)
@get("/GID")
def get_gid():
        renewList=[]
        unknownList=[]
        closeList=[]
        deleteList=[]
        queryvars = parse_qs(request.query_string)
        print type(queryvars)
        for htmlvar in queryvars:
                htmlvalue=queryvars.get(htmlvar)[0]
                print htmlvalue
                if htmlvar.endswith("radio"):
                    if htmlvalue == "known":
                        print(htmlvar.split("radio",1)[0] + " is renew")
                        renewList.append(htmlvar.split("radio",1)[0])
                        #print("This is where you do things with confirmed accounts")
                    elif htmlvalue == "unknown":
                        print(htmlvar.split("radio",1)[0] + "is unknown")
                        unknownList.append(htmlvar.split("radio",1)[0])
                       # print("This is where you do things with unknown accounts")
		    elif htmlvalue == "close":
                        print(htmlvar.split("radio",1)[0] + "is closed")
                        closeList.append(htmlvar.split("radio",1)[0])
                       # print("This is where you do things with unknown accounts")

                elif htmlvar.endswith("check"):
                        print("Deletion of " + htmlvar.split("check",1)[0] + " is confirmed")
                        deleteList.append(htmlvar.split("check",1)[0])
                       #print("This is where you do things with deleted accounts")
                else: #ignore the 'submit'
                    pass
        for i in deleteList:
                if i in closeList:
                        closeList.remove(i)
        #Do things with your csvs using these lists, confirmedList unknownList deleteList
        print "RENEW LIST " + str(renewList)
        print "UNKNOWN LIST " + str(unknownList)
        print "DELETE LIST " + str(deleteList)
	print "CLOSE LIST" + str(closeList)
run(host='localhost', port=8081, debug=True)
