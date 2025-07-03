<?php
$link=mysqli_connect("iez085.ieda.ust.hk", "ytsang", "20705565");
mysqli_select_db($link, "ytsang_db");
$action = $_POST['action'];
$counter = 0;
//echo "$action";
//action: 0-Create; 1-Deactivate; 2-Update; 3-Summary
?>


<!DOCTYPE html>
<html>
    <head>
        <title>Project Summary Report</title>
        <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
    </head>
    <h3><font color=blue>
    Provide Project Summary Report
    </h3> </font> 
<body>
    <?php

        
    ?>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script>
        $(document).ready(function(){
            $("#form1 #select-all").click(function(){
                $("#form1 input[type='checkbox']").prop('checked',this.checked);
            });
        });
    </script>
    <form id="form1" method="POST">
    <table border="1" cellpadding="5" cellspacing=0>
        <th></th>
        <th>Project Number</th>
        <th>Project Name</th>
        <?php
            $stmt=mysqli_query($link, "SELECT * FROM project");
            //$stmt->execute();
            while($row = mysqli_fetch_array($stmt)){
                $projNoSel = $row['Pnumber'];
                $projNameSel = $row['Pname'];
                echo "<tr><td><input type='checkbox' name='check[]' value='$projNoSel'/></td>
                    <td>$projNoSel</d><td>$projNameSel</td></tr>";
            }
        ?>
        <tr><td><input type="checkbox" id="select-all"></td>
        <td><input type='submit' name='query' value='Query'/></td>
        <td><input type=reset value="Reset Input"></td></tr>
    </table>
    </p>
    </form>
    <a href="http://ytsang.student.ust.hk/ieda3300/mini-project.html"><button>Return Home</button></a>
    <hr>

    <?php
    if(isset($_POST['query'])){
        
    ?>
    <table border="1" cellpadding="5" cellspacing=0>
        <th>Project Number</th>
        <th>Project Name</th>
        <th>Ctrl Dept</th>
        <th>Manager</th>
        <th>No of Employees</th>
        <th>Total Working Hours</th>
        <th>Total Personnel Cost</th>
        <?php
            foreach($_POST['check'] as $value)
            {
            $sql="SELECT t5.Pnumber, t5.Pname, t5.Dname, t5.managerName, t5.employeeNo, IFNULL(t5.projHours,0) AS projHours, IFNULL(SUM(t6.projEmplPartialCost),0) AS projSumCost FROM
            (SELECT t3.Pnumber, t3.Pname, t3.employeeNo, t3.projHours, t4.Ssn, t4.managerName, t3.Dname FROM
            (SELECT project.Pnumber, project.Pname, COUNT(works_on.Essn) As employeeNo, SUM(works_on.Hours) AS projHours, department.Mgr_ssn, department.Dname
            FROM employee
            JOIN works_on ON employee.Ssn = works_on.Essn
            RIGHT JOIN PROJECT ON works_on.Pno = project.Pnumber
            JOIN department ON project.Dnum = department.Dnumber
            GROUP BY project.Pnumber) AS t3
            JOIN
            (SELECT employee.Ssn, CONCAT(employee.Fname,' ',employee.Minit,' ',employee.Lname) As managerName FROM employee) AS  t4
            ON t3.Mgr_ssn = t4.Ssn
            GROUP BY Pnumber
            ORDER BY Pnumber) AS t5
            LEFT JOIN
            (SELECT DISTINCT Pno, t1.Ssn, employeeFullName, projectHours, totalHours, ROUND((t1.projectHours*t1.Salary/t2.totalHours),2) AS projEmplPartialCost FROM
            (SELECT works_on.Pno, employee.Ssn, CONCAT(employee.Fname,' ', employee.Minit,' ',employee.Lname) AS employeeFullName, SUM(works_on.Hours) AS projectHours, employee.Salary FROM employee
            JOIN works_on ON employee.Ssn = works_on.Essn
            GROUP BY works_on.Pno, employee.Ssn) AS t1
            JOIN
            (SELECT works_on.Essn, SUM(works_on.Hours) AS totalHours FROM works_on 
            GROUP BY works_on.Essn) AS t2
            WHERE t1.Ssn = t2.Essn
            ORDER BY Pno, t1.Ssn) AS t6
            ON t5.Pnumber = t6.Pno
            WHERE Pnumber = $value
            GROUP BY t5.Pnumber
            ORDER BY t5.Pnumber";
            $stmt=mysqli_query($link, $sql);

            while($row = mysqli_fetch_array($stmt)){
                ?><tr>
                <td><?php echo $row["Pnumber"];?></td>
                <td><?php echo $row["Pname"];?></td>
                <td><?php echo $row["Dname"];?></td>
                <td><?php echo $row["managerName"];?></td>
                <td><?php echo $row["employeeNo"];?></td>
                <td><?php echo $row["projHours"];?></td>
                <td><?php echo $row["projSumCost"];?></td>
                </tr><?php
            }
        }
        ?>
    </table>
    <?php
    }
    ?>
</body>
</html>
