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
//learstatcache();
//header("Cache-Control: no-cache, must-revalidate");
//header("Expires: Mon, 26 Jul 1997 05:00:00 GMT");
$counter = 0;
$projNoSel;$projNameSel = "";
$projEssn = 0;$projHours = 0;
$emplNoSel = 0;$emplFNameSel = 0;$emplLNameSel = 0;
$hide = 2;
//header("Content-Type: application/xml; charset=utf-8");
// action: 0-Create; 1-Deactivate; 2-Update; 3-Summary
?>
<!DOCTYPE html>
<html>
    <head>
        <title>Update Worker Information</title>
        <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
    </head>
    <h3><font color=blue>
    Update Worker Information for a Project
    </h3> </font> 
<body>
    <table border=1>
    <style>form {border: 3px solod black;}</style>

    <table> 
    <form name="form0" action="" method="get"> 
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
            <td><?php echo "<input type='radio' name='selected_1' value='".$counter."'/>";?></td>
            </tr>
            <?php $counter++; ?>
            <?php
        }
        ?>
    <tr> 
        <td><input name="retrieve_btn" type="submit" value="Retrieve"/></td>
        <td><input type=reset value="Reset Input"></td>
        </form>
        <td><a href="http://ytsang.student.ust.hk/ieda3300/mini-project.html"><button>Return Home</button></a></td>
    </tr> 

</table> 
<?php 
    $value=$_GET['selected_1'];
    $res=mysqli_query($link, "select * from project");
    while($row=mysqli_fetch_array($res))
    { 
        $projNoSel = $row["Pnumber"];
        $projNameSel = $row["Pname"];
        if ($value==0){break;}
        $value--;
    }

    $counter = 0;

    if(isset($_REQUEST['retrieve_btn'])){$hide = 0;}
    if (isset($hide)){
        echo "<hr><br>Employee(s) in Project $projNoSel $projNameSel: <br>";
        $res=mysqli_query($link, "select * from works_on join employee on employee.Ssn = works_on.Essn 
            where works_on.Pno = $projNoSel");
        ?>
        <table>
        <form name="form1" action="" method="post">
        <tr> 
        <td> SSN</td><td>First Name</td><td>Last Name</td><td>Hours</td>
        </tr>   
        <?php
        while($row=mysqli_fetch_array($res))
        {
            ?>
            <tr>
            <td><?php echo $row["Ssn"];?></td>
            <td><?php echo $row["Fname"];?></td>
            <td><?php echo $row["Lname"];?></td>
            <td><?php echo $row["Hours"];?></td>
            <td><?php echo "<input type='radio' name='selected_2' value='".$counter."'/>";?></td>
            </tr>
            <?php $counter++; ?>
            <?php
        }
        $value=$_POST['selected_2'];
        $res=mysqli_query($link, "select * from works_on join employee on employee.Ssn = works_on.Essn 
        where works_on.Pno = $projNoSel");
        while($row=mysqli_fetch_array($res))
        { 
            $emplNoSel = $row["Ssn"];
            $emplFNameSel = $row["Fname"];
            $emplLNameSel = $row["Lname"];
            if ($value==0){break;}
            $value--;
        }
        ?>
        <tr> 
        <td><input name="delete_btn" type="submit" value="Remove"/></td>
        <td><input name="modify_btn" type="submit" value="Modify"/></td>
        <td><input name="insert_btn" type="submit" value="Insert"/></td>
        <td><input name="reset_btn_2" type=reset value="Reset Input"></td>
        </tr> 
        </table>
        </form>
        <?php

        
    }
    
    if(isset($_REQUEST['delete_btn']))
    {   
        $sql = "DELETE FROM `works_on` WHERE Essn='$emplNoSel'&& Pno=$projNoSel";
        mysqli_query($link, $sql);     
        echo "<hr><br>Selected Employee in Project $projNoSel $projNameSel: <br>";
        echo "<table><tr><td>Employee Number</td><td>Name:</td></tr>"; 
        echo "<tr><td>$emplNoSel</td><td>$emplFNameSel $emplLNameSel</td></tr></table><br>";
        echo "Action: Remove! <br><hr>";
    }


    if(isset($_REQUEST['modify_btn']))
    {   
        echo "<hr><br>Selected Employee in Project $projNoSel $projNameSel: <br>";
        echo "<table><tr><td>Employee Number</td><td>$emplNoSel</td></tr>"; 
        echo "<tr><td>Name</td><td>$emplFNameSel $emplLNameSel</td></tr>";
        $res=mysqli_query($link, "SELECT Ssn, Fname, Lname FROM `employee` where employee.Ssn IN
            (SELECT Essn from works_on where works_on.Pno = $projNoSel and works_on.Essn = $emplNoSel)");
        ?>
          
        <form name="form3" action="" method="post">
        <table>
        <tr><td>Please Enter the New Working Hour for the Employee:</td>

        <td>
            <input name="modify_hour_txtfield" size=45 type="txt" placeholder="Please enter the working hour for the selected employee:" required/>
        </td></tr>
        <tr><td>Please Confirm the Modification by Choosing "Yes":</td>
        <td>
            <select name="employeeSSN_M"><option disabled selected>-- Select Employee -- </option>  
            <?php
            while($row=mysqli_fetch_array($res))
            {    
                echo "<option value='".$row['Ssn']."'>" .Yes."</option>"; 
                //$employeeSSN_M = $row['Ssn'];
                //echo "<input type='radio' name='employeeSSN_M' value='".$row['Ssn']."'/>";
            }
            ?>
            
            </select>
        </td></tr>
        <tr>
            <td><input name="modify_2_btn" type="submit" value="Update"/></td>
            <td><input name="reset_btn_4" type=reset value="Reset Input"></td>
        </td></table>
            </form>
             
        
        <?php        

    }
    
    if(isset($_POST['modify_2_btn'])){
        $selectedSSN = $_POST['employeeSSN_M'];
        if(!empty($_POST['modify_hour_txtfield'])){
            $selectedSSN = $_POST['employeeSSN_M'];
            //echo "selected SSN Value = $selectedSSN<br>";
            $sql = "SELECT Fname, Lname from employee where employee.Ssn = $selectedSSN";
            $res = mysqli_query($link, $sql);
            while($row=mysqli_fetch_array($res))
            { 
                $emplFNameSel = $row["Fname"];
                $emplLNameSel = $row["Lname"];
            }
            $modify_hour_txtfield = $_POST['modify_hour_txtfield'];
            $sql = "UPDATE `works_on` SET Hours = $modify_hour_txtfield WHERE works_on.Essn = $selectedSSN 
                and works_on.Pno = $projNoSel";
            mysqli_query($link, $sql);
            echo "<hr>$emplFNameSel $emplLNameSel($selectedSSN) has been added successfully.<br>";
            echo "He/She is assigned for $modify_hour_txtfield hours a week in project no: $projNoSel.<br>";
            //echo "Project No: $projNoSel <br>";
        }
        //elseif($selectedSSN == '1'){echo "Please Reconfirm the selected employee."；}
        else echo"Invalid Input! Please try again.";

    }



    $emplNoSel = 0;$emplFNameSel='';$emplLNameSel='';
    if(isset($_REQUEST['insert_btn']))
    { 
        echo "<hr>List of Employees that Not Currently Working in Project $projNoSel $projNameSel:<br>";
        $res=mysqli_query($link, "SELECT Ssn, Fname, Lname FROM `employee` where employee.Ssn NOT IN
            (SELECT Essn from works_on where works_on.Pno = $projNoSel)");
        ?>
          
        <form name="form2" action="" method="post">
        <table><tr><td>Employees Available to be Assigned to This Project:</td>
        <td>
            <select name="employeeSSN"><option disabled selected>-- Select Employee -- </option>  
            <?php
            while($row=mysqli_fetch_array($res))
            {    
                echo "<option value='".$row['Ssn']."'>" . $row['Ssn'].": ".$row['Fname']." ".$row['Lname']."</option>"; 
            }
            ?>
            <?php $counter++; ?>
            </select>
        </td></tr>
        <tr><td>Please Selected the Employee to be Assigned:</td>

        <td>
            <input name="update_hour_txtfield" size=45 type="txt" placeholder="Please enter the working hour for the selected employee:" required/>
        </td></tr>
        <tr>
            <td><input name="update_btn" type="submit" value="Update"/></td>
            <td><input name="reset_btn_3" type=reset value="Reset Input"></td>
        </td></table>
            </form>
             
        
        <?php
    }
    if(isset($_POST['update_btn'])){
        if(!empty($_POST['employeeSSN'])&&!empty($_POST['update_hour_txtfield'])){
            $selected = $_POST['employeeSSN'];
            $sql = "SELECT Fname, Lname from employee where employee.Ssn = $selected";
            $res = mysqli_query($link, $sql);
            while($row=mysqli_fetch_array($res))
            { 
                $emplFNameSel = $row["Fname"];
                $emplLNameSel = $row["Lname"];
            }
            $update_hour_txtfield = $_POST['update_hour_txtfield'];
            $sql = "INSERT INTO `works_on` (`Essn`, `Pno`, `Hours`) VALUES ('$selected', '$projNoSel', '$update_hour_txtfield')";
            mysqli_query($link, $sql);
            echo "<hr>$emplFNameSel $emplLNameSel($selected) has been added successfully.<br>";
            echo "He/She is assigned for $update_hour_txtfield hours a week.<br>";
            }
        else echo"Invalid Input! Please try again.";
    }
    
    ?>

</body>
</html>
