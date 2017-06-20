<!doctype html>
<script language="javascript">
function checkRadio(id){

//Checks if radio button is clicked, radio buttons are named <user>del,<user>close,<user>unk, and user<renew>
//<user>hide is a div used to conceal the checkbox, if close account is checked, it is revealed, else it is hidden and unchecked


   var delstr=id.concat("del");
   var renewstr=id.concat("renew");
   var closestr=id.concat("close");
   var unkstr=id.concat("unk");
   var hidestr=id.concat("hide");
//   var renewbutton=document.getElementById(renewstr);
   var delbutton=document.getElementById(delstr);
   var closebutton=document.getElementById(closestr);
//   var unkbutton=document.getElementById(unkstr);
   var divbox=document.getElementById(hidestr);
    if (closebutton.checked) {
        divbox.style.display='inline';
        }
    else{
        divbox.style.display='none';
        delbutton.checked = false;
        }
}
</script>

<html>
 <head>
  <link rel="stylesheet" type="text/css" href="/css/css.css"/>
 </head>
 <body>
  <div class="main">
   <div class="sub_input">
     <div class="header">
      <h1>Known Accounts</h1>
      <form method="get" action="/GID">
        Professor GID: <input name="USER_IN" id="USER_IN" pattern="[A-Za-z]{1,}" class="text_input" type="text">
        <input type="submit" id="input">
      </form>
      <h3 style="display: {{error}}">Sorry, that groupname doesn't exist, try again?</h3>
     </div>
     <form method="get" action="/end">
       <div class="mass">
         <div style="display: {{finish}}">
           <br> Accounts set for renewal: {{', '.join(renewList)}} </br>
           <br> Accounts set for deletion: {{', '.join(deleteList)}} </br>
           <br> Accounts set to be closed:{{', '.join(closeList)}} </br>
           <br> Accounts that are unknown:{{', '.join(unknownList)}} </br>
         </div>
         <div class="users" style="display: {{accounts}}">
           <div class="title">
             <h2>Accounts Attached to GID</h2>
           </div>
           % x=0
           % for i in User_list:
          <!-- % radioname = "User" + str(x) -->
             %user=User_list[x]
             {{User_list[x]}}
             %dofunc="checkRadio("+'"' + user + '"' +")"
           <div class"accounts">
              <input type="radio" id={{user}}renew name={{user}}radio value="known" onclick={{dofunc}}>Renew Account
              <input type="radio" id={{user}}close name={{user}}radio value="close" onclick={{dofunc}} >Close Account
	            <input type="radio" id={{user}}unk name={{user}}radio value="unknown" onclick={{dofunc}} checked>Unknown account
              <span id={{user}}hide style="display: none"><input type="checkbox" id={{user}}del name={{user}}check value="delete">Delete Account</span>
           </div>
           <br><br>
           % x+=1
           % end
           <input type="submit" name=submit>
         </div>
     </form>
   </div>
   </div>
 </div>
 </body>
</html>
