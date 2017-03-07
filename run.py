from bottle import *

#Static
@route('/<filename:path>')
def send_static(filename):
    return static_file(filename, root='static/')

#Template
@route('/')
def main():
    return template('index.tpl')

@post("/format")
def post_gid():
    gid = request.forms.get("USER_IN")
    radioname = request.forms.get("radioname")
    print gid
    print radioname
    redirect('/GID?gid={gid}'.format(gid=gid))
    redirect('/submit?radioname={radioname}'.format(radioname=radioname))

@get("/GID")
def get_gid():
    USER_IN = request.query.get("gid") or ""
    return template('accounts.tpl', USER_IN=USER_IN)

@get("/submit")
def get_user():
    User_list = request.query.get("radioname") or ""
    print User_list
    x = 0
    for i in User_list:
        if User_list(x) == "Yes":
            target = open(Aprroved.txt, 'w')
            target.write({{GID}} + "/t" + {{User_list[x]}})
            target.close()
        elif User_list(x) == "Unknown":
            target = open(Unknown.txt, 'w')
            target.write({{GID}} + "/t" + {{User_list[x]}})
            target.close()
        elif User_list(x) == "No":
            target = open(Denied.txt, 'w')
            target.write({{GID}} + "/t" + {{User_list[x]}})
            target.close()
            x+=1
    end


run(host='localhost', port=8081, debug=True)
