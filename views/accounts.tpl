<!doctype html>
<html>
 <head>
  <link rel="stylesheet" type="text/css" href="/css/css.css"/>
 </head>
 <body>
  <div class="main">
   <div class="sub_input">
     <div class="header">
      <h1>Known Accounts</h1>
     </div>
     <form>
       % target_Groups_list = [ line for line in open('Files/Groups') if USER_IN in line]
       % target_Groups_str = ''.join(target_Groups_list)
       % Group, trash, GID, trash = target_Groups_str.split(":", 3)
       <div class="mass">
         % x = 0
         % Name = ""
         % target_Passwd_list = [ line for line in open('Files/Passwd') if GID in line ]
         % User_list = [i.split(':', 1)[0] for i in target_Passwd_list]
         % for i in User_list:
         % x+=1
         % end
         <div class="users">
           <h2>Accounts Attached to GID</h2>
           % x = 0
           % for i in User_list:
           % radioname = "User" + str(x)
           {{User_list[x]}}
           <input type="radio" name={{radioname}} value="Yes">Known account
           <input type="radio" name={{radioname}}  value="Unknown">Unknown account
           <input type="radio" name={{radioname}}  value="No">Not my account
           <br><br>
           % x+=1
           % end
           <input type="submit">
         </div>
     </form>
   </div>
 </div>
 </body>
</html>
