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
$counter = 0;
$projNoSel = 0;$projNameSel = "";
$projEssn = 0;$projHours = 0;
//header("Content-Type: application/xml; charset=utf-8");
// action: 0-Create; 1-Deactivate; 2-Update; 3-Summary
?>
<!DOCTYPE html>
<html>
    <head>
        <title>Deactivate Project</title>
        <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
    </head>
    <h3><font color=blue>
        Deactive an Existing Project
    </h3> </font> 
<body>
    <table border=1>
    <style>form {border: 3px solod black;}</style>

    <table> 
    <form action="" method="post"> 
    <tr> 
        <td> Project Number</td><td>Project Name</td><td>Select</td>
    </tr>  
        <?php
        $res=mysqli_query($link, "select * from project");
        while($row=mysqli_fetch_array($res))
        {
            ?>
            <tr>
            <td><?php echo $row["Pnumber"];?></td>
            <td><?php echo $row["Pname"];?></td>
            <td><?php echo "<input type='radio' name='selected' value='".$counter."'/>";?></td>
            </tr>
            <?php $counter++; ?>
            <?php
        }
        ?>
    <tr> 
        <td><input name="submit_btn" type="submit" value="Deactivate"/></td>
        <td><input type=reset value="Reset Input"></td>
        </form> 
        <td><a href="http://ytsang.student.ust.hk/ieda3300/mini-project.html"><button>Return Home</button></a></td>
    </tr> 

</table> 
<?php 
    $value=$_POST['selected'];
    $res=mysqli_query($link, "select * from project");
    while($row=mysqli_fetch_array($res))
    { 
        $projNoSel = $row["Pnumber"];
        $projNameSel = $row["Pname"];
        if ($value==0){break;}
        $value--;
    }
?>
<hr>
<?php
    //$projNo = $_POST['projNo'];
    //$name = $_POST['name'];

    if(isset($_REQUEST['submit_btn']))
    {   
        echo "Project $projNoSel $projNameSel has been deactivated successfully! <br>";
        //$res=mysqli_query($link, "SELECT * FROM `works_on` WHERE Pno =$projNoSel;");
        //while($row=mysqli_fetch_array($res))
        //{
             
        //    $projEssn = $row["Essn"];
        //    $projHours = $row["Hours"];
        //    echo "<tr><td>$projNoSel</td><td>$projEssn</td><td>$projHours</td></tr>";
        $SQL = "DELETE FROM `works_on` WHERE `works_on`.`Pno` = $projNoSel;";
        mysqli_query($link, $SQL);
        $SQL = "DELETE FROM `project` WHERE `project`.`Pnumber` = $projNoSel;";
        mysqli_query($link, $SQL);
        //}
        //echo "</table>";
    }
    
    ?>

</body>
</html>
