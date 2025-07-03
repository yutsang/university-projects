<?php
//Initialisation
$host = "iez085.ieda.ust.hk";
$usrname = "ytsang";
$usrpw = "20705565";
$dbname = "ytsang_db";
//Initialisation End

$link=mysqli_connect($host, $usrname, $usrpw);
mysqli_select_db($link, $dbname);
$action = $_POST['action'];
clearstatcache();
header("Cache-Control: no-cache, must-revalidate");
header("Expires: Mon, 26 Jul 1997 05:00:00 GMT");
//header("Content-Type: application/xml; charset=utf-8");
// action: 0-Create; 1-Deactivate; 2-Update; 3-Summary
?>
<!DOCTYPE html>
<html>
    <head>
        <title>Create Project</title>
        <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
    </head>
    <h3><font color=blue>
        Insert a New Project
    </h3> </font> 
<body>
    <table border=1>
    <style>form {border: 3px solod black;}</style>

    <table> 
    <form action="" method="post"> 
    <tr> 
        <td> Project Number: </td><td><input type="text" name="projNo"></td> 
    </tr> 
    <tr> 
        <td> Name: </td><td><input type="text" name="name"></td> 
    </tr> 
    <tr> 
        <td> Location: </td><td><input type="text" name="location"></td> 
    </tr> 
    <tr> 
        <td> Ctrl Dept: </td><td><select name="ctrlDept" value="<?php echo $ctrlDept;?>">
        <?php
        $res=mysqli_query($link, "select * from department");
        while($row=mysqli_fetch_array($res))
        {
            ?>
            <option><?php echo $row["Dnumber"]; ?></option>
            <?php
        }
        ?></select></td> 
    </tr> 
    <tr> 
        <td><input name="submit_btn" type="submit"/></td>
        <td><input type=reset value="Reset Input"></td></form>
        <td><a href="http://ytsang.student.ust.hk/ieda3300/mini-project.html"><button>Return Home</button></a></td>
    </tr> 
</table> 

<?php
    $projNo = $_POST['projNo'];
    $name = $_POST['name'];
    $location = $_POST['location'];
    $ctrlDept = $_POST['ctrlDept'];
    $SQL = "INSERT INTO `project` (`Pname`, `Pnumber`, `Plocation`, `Dnum`) 
        VALUES ('$name', '$projNo', '$location', '$ctrlDept')";
    echo "<p><hr><p>";

    if((isset($_REQUEST['submit_btn']))&&($name!=null)&&($location!=null)&&($projNo!=null))
    {
        mysqli_query($link, $SQL); 
        echo "\nSubmited Successfully! <br>Submit info:";
        echo "<table border:3px><tr><td>Project Number:</td><td>$projNo </td></tr>";
        echo "<tr><td>Name:</td><td>$name</td></tr>";
        echo "<tr><td>Project Location:</td><td>$location</td></tr>";
        echo "<tr><td>Ctrl Dept Num:</td><td>$ctrlDept</td></tr></table>";
        }
        else if((isset($_REQUEST['submit_btn']))&&(($name==null)||($location==null)||($projNo==null))){
            echo "Please input the valid data.\n";
        }
    ?>

</body>
</html>
