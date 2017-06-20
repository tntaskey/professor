#!/bin/python
import bottle as bottle
from bottle import *
import os
import urlparse # if we're pre-2.6, this will not include parse_qs
global User_list
global renewList
global unknownList
global closeList
global deleteList
global gid
global uid
User_list=[]
renewList=[]
unknownList=[]
closeList=[]
deleteList=[]
uid=[]

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
@route('/',method=["GET","POST"])
def post_home():
    accounts = "none";
    error = "none";
    finish = "none";
    USER_IN=request.query.get("USER_IN")
    return template('index.tpl', renewList=renewList, unknownList=unknownList, deleteList=deleteList, closeList=closeList, User_list=User_list, accounts=accounts, error=error, finish=finish)

@get("/GID")
def get_gid():
    User_list[:] = []
    renewList[:] = []
    unknownList[:] = []
    closeList[:] = []
    deleteList[:] = []
    USER_IN=request.query.get("USER_IN")
    global gid
    global uid
    finish = "none";
    for i in open('Files/Groups'):
            if USER_IN == i.split(":",3)[0]:
                gid=i.split(":",3)[2]
    #check Passwd for user name, if GID matchs pgroup GID then add user to userList
    for i in open('Files/Passwd'):
            if gid == i.split(":",5)[3]:
                uid.append(i.split(":",5)[2])
                User_list.append(i.split(":",5)[0])
    if gid == "":
            error = "block";
            accounts = "none";
            return template('index.tpl', renewList=renewList, unknownList=unknownList, deleteList=deleteList, closeList=closeList, USER_IN=USER_IN, User_list=User_list, error=error, accounts=accounts, finish=finish)
    else:
            accounts = "block";
            error = "none";
            return template('index.tpl', renewList=renewList, unknownList=unknownList, deleteList=deleteList, closeList=closeList, User_list=User_list, accounts=accounts, error=error, finish=finish)

@get("/end")
def post_gid():
        finish = "block";
        accounts = "none";
        error = "none";
        queryvars = parse_qs(request.query_string)
        print type(queryvars)
        for htmlvar in queryvars:
                htmlvalue=queryvars.get(htmlvar)[0]
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

        target = open("decisions.csv", 'a+')
        for htmlvar in queryvars:
            RadioUser = htmlvar.split("radio",1)[0]
            CheckUser = htmlvar.split("check",1)[0]

        x=0
        for i in queryvars:
            if RadioUser[x] == i.split(",",3)[2] or CheckUser[x] == i.split(",",3)[2]:
                pass
            else:
                if htmlvar.split("radio",1)[0] in renewList:
                    target.write(gid + "," + uid[x] + "," + RadioUser + "," + "renew,false" + '\n')
                elif htmlvar.split("radio",1)[0] in unknownList:
                    target.write(gid + "," + uid[x] + "," + RadioUser + "," + "unknown,false" + '\n')
                elif htmlvar.split("radio",1)[0] in closeList:
                    target.write(gid + "," + uid[x] + "," + RadioUser + "," + "close,false" + '\n')
                elif htmlvar.split("check",1)[0] in deleteList:
                    target.write(gid + "," + uid[x] + "," + CheckUser + "," + "close,true" + '\n')
                else:
                    pass
            x+=1
        target.close

        #Do things with your csvs using these lists, confirmedList unknownList deleteList
        print "RENEW LIST " + str(renewList)
        print "UNKNOWN LIST " + str(unknownList)
        print "DELETE LIST " + str(deleteList)
     	print "CLOSE LIST" + str(closeList)
        return template('index.tpl', renewList=renewList, unknownList=unknownList, deleteList=deleteList, closeList=closeList, User_list=User_list, error=error, accounts=accounts, finish=finish)
run(host='localhost', port=8081, debug=True)
