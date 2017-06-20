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
     <div>
        <form>
   <br> Accounts set for renewal: {{', '.join(renewList)}} </br>
   <br> Accounts set for deletion: {{', '.join(deleteList)}} </br>
   <br> Accounts set to be closed:{{', '.join(closeList)}} </br>
   <br> Accounts that are unknown:{{', '.join(unknownList)}} </br>
    </form>
     </div>
   </div>
 </div>
 </body>
</html>
