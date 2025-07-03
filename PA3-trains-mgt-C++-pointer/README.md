## Train Optimization Program User Handbook
# Background
This program implements a linked list to represent a train with multiple cars carrying cargo. The goal of the program is to allow the user to perform various operations on the train, including:  

* Adding/removing cars
* Swapping the position of cars
* Sorting the cars by cargo type
* Loading/unloading cargo from cars
* Printing statistics and dividing the train by cargo type
* Optimizing the train to maximize cargo within a given upper bound  

The program utilizes pointers and dynamic memory allocation to implement the linked list. It is a good example of using algorithms and data structures to model a real-world problem.  

* How to Use
To use the program, first compile all the files by running:

g++ given.cpp main.cpp todo.cpp -o pa3  

Then run the executable:

./pa3  

The program will prompt the user to select one of the operations on the train. Enter the number of the test case you want to run:  

* Add a new car
* Remove a car
* Swap two cars
* Sort the train by cargo type
* Load cargo onto a car
* Unload cargo from a car
* Print cargo statistics
* Divide the train by cargo type
* Optimize the train  

The program will then prompt you to enter any required parameters for the operation you selected and execute the operation on the train, printing the results.  
For example, if you select option 1 to add a new car, it will prompt you to enter the cargo type and amount for the new car. If you select option 3 to swap cars, it will prompt you for the indices of the two cars to swap.  

Some notes:  
* The train begins empty, you must first add some cars before performing other operations.  
* When removing a car, loading/unloading cargo, or swapping cars, enter the index of the car starting from 0.  
* For option 9 (optimize train), also enter the maximum total cargo for the optimized train.  
* You can run multiple options in one execution of the program. The train state will persist between operations.
