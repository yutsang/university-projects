# The-Magic-Array
<p align="center">
<img width="225" alt="Screenshot 2022-10-01 at 23 31 42" src="https://user-images.githubusercontent.com/19373417/193416789-5d2c6bda-77f3-4fb2-b7c3-dcc5f0083022.png">
</p>

# Introduction
Most of you may have heard of or played with the **Rubik's Cube** or the **Magic Cube**, a 3D combination puzzle invented in 1974 by Hungarian sculptor and professor of architecture Ernő Rubik. The classic **Magic Cube** is a cube-shaped device made up of 26 small cubes that rotate on a central axis; nine colored cube faces, in three rows of three each, form each side of the cube. When you rotate a plane on an axis, all cube faces on the same plane will rotate together. To solve the puzzle, you are supposed to perform a series of rotations to transform it into the target cube face combination—one among 43 quintillion possible ones.  

While the original **Magic Cube** puzzle is difficult to solve, in this assignment we are going to consider a simpler 1D version of the **Magic Cube**, referred to as the **Magic Array**. The **Magic Array** is implemented using a 1D integer array in C++ . You are supposed to first implement the basic rotation action, which will reverse the order of all elements in the specified subarray. Then, based on the rotation actions, you are supposed to implement three advanced actions , i.e. swap, sort and transform.
<p align="center">
<img width="621" alt="Screenshot 2022-10-01 at 23 38 53" src="https://user-images.githubusercontent.com/19373417/193417050-177a9cc0-ce4c-4273-a516-d81f99ccbf62.png">
</p>

# Definition
## Basic Action: Rotation
<p align="center">
<img width="624" alt="Screenshot 2022-10-01 at 23 41 03" src="https://user-images.githubusercontent.com/19373417/193417127-37658d9c-a800-4f38-96e6-fa577c0759cf.png">
</p>

To play with the **Magic Array**, you are not supposed to directly change the order or value of the elements (just like you are not allowed to solve the **Magic Cube** by disassembling and reassembling). What you can do is to perform rotation on the array. Given the array we have defined, the rotate action is defined as the following two steps:  
1. Choose an integer k where  
  <img src = "https://latex.codecogs.com/svg.image?k\in&space;\left&space;[&space;0,&space;n-1&space;\right&space;]">  
2. Reverse the subarray 
  <img src = "https://latex.codecogs.com/svg.image?\bg{white}\left\{a_{i}\mid&space;i=0,&space;1,&space;...&space;k\right\}">

## Advanced Actions
By performing a series of rotate actions, we can further implement the following complex actions: swap, sort and transform.  

### Swap by Rotation
One rotation action will change the order of all elements in the specified subarray. So, it is tricky if we only want to swap the position of two elements without affecting the order of the others. Denoting the left and right index of the two elements to be swapped as L and R, the swap in a general case can be done by performing the following six rotations:  
  1. k = L - 1
  2. k = L
  3. k = R
  4. k = R - L - 1
  5. k = R - L - 2
  6. k = R - 1
  
A visualization:  
<img width="489" alt="Screenshot 2022-10-01 at 23 57 18" src="https://user-images.githubusercontent.com/19373417/193417744-f0ca3cc9-200e-42f6-b5b0-14fae46b501d.png">  
While the above six steps give a solution to the general cases, there are some special boundary cases that the red/green/purple parts in the figure above are missing, e.g., when L = 0, there won't be the red part. You are supposed to consider those special cases in this assignment.  

## Sort by Rotation
Sorting is another important action when we play with the **Magic Array** with rotations. Rotations can be used to sort all elements in the **Magic Array** too.
For example, Bubble Sort is a well-known sorting algorithm (you can find detailed information [here](https://en.wikipedia.org/wiki/Bubble_sort), and a visualization [here](https://visualgo.net/en/sorting)). To sort a list, Bubble Sort repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted.
We can adapt Bubble Sort to our **Magic Array** by substituting the swap action in the original algorithm with the "swap by rotation" action we have defined.  

## Transform by Rotation
Using the "sort by rotation" action, we can easily turn a given array into its sorted form by a series of rotations. Here we extend the sort action to a more general case, that we refer to as the "transform by rotation" action. Given two **Magic Array** source and target, if target has the same elements as source but in a different ordering, there exists a "transform by rotation" action, which uses a series of rotations to turn the source array into the target array.  
<img width="491" alt="Screenshot 2022-10-02 at 00 03 00" src="https://user-images.githubusercontent.com/19373417/193418005-052cede1-62f9-40f5-a7be-5e4c3c783f24.png">  

#Tasks
Before you start to do the tasks, please carefully read through the Definitions sections.  

We have provided a skeleton code for you, and you must complete the following tasks based on that skeleton code. You are encouraged to complete those tasks in the given order. And you are encouraged to reuse the functions you have already implemented. You can also write your own helper functions, but please make sure the function you call is bug-free, or you may lose marks in all tasks that call this function.
In this assignment, you can assume the length of the **Magic Array** is **NO MORE THAN** 16. And the length of the rotation series is **NO MORE THAN** 1024.  
Two global constants have been defined in the skeleton code for you to use (You cannot change those defined global variables):

<img width="482" alt="Screenshot 2022-10-02 at 00 03 57" src="https://user-images.githubusercontent.com/19373417/193418048-2da0788b-8f71-4e59-b9c9-a77e8737f408.png">

## Task 1
**Rotate**

  int rotate(int arr[], int arrLen, int k)  

* Parameters:  
  * arr: the array to be rotated
  * arrLen: the length of arr
  * k: the integer specifying the subarray to be rotated
* Description: 
  * If k is not in [0, arrLen - 1]:
    * print "Error: Index k is out of range." to the screen ( standard output ) with an ending endl 
    * return -1
  * Else 
    * reverse the subarray of arr
    * return 0
    
* Example:
1.  * arr =[9,1,3,5,4], arrLen =5, k =3
    * After function call:    
      * arr =[5,3,1,9,4]    
      * return=0
2.  * arr =[9,1,3,5,4], arrLen =5, k =6 
    * After function call:   
      * print "Error: Index k is out of range." 
      * return=-1

## Task 2
**Swap two elements of the array by rotation without affecting the order of the other elements, and record the ks used in the rotations.**

  int swapAndRecord(int arr[], int arrLen, int indexA, int indexB, int rotations[], int
&rotationLen)。

* Parameters: 
  * arr: the array
  * arrLen: length of arr
  * indexA and indexB: he index of the two elements to be swapped
  * rotations: rotations to swap the two elements as required
  * rotationLen: length of rotations
  
* Description:
  * If indexA or indexB is not in [0, arrLen - 1]:
    * Print "Error: Index out of range." to the screen ( standard output ) with an ending endl
    * return -1
  * Else:
    * Swap the elements at indexA and indexB of arr by rotations
    * store the k values of the rotations performed in rotations
    * store the count of rotations performed in rotationLen
    * return 0  
  The answers are **NOT UNIQUE**. Any series of rotations that can swap the two elements are accepted, but the length of the rotation series should be **NO MORE THAN** MAX_ROTATIONS  
  
* Example:
1.  * arr =[1,3,5,7,2,4,6], arrLen =7, indexA =5, indexB =2
    * After function call:    
      * arr =[1,3,4,7,2,5,6]
      * rotations =[1,2,5,2,1,4]
      * rotationLen =6
      * return=0
2.  * arr =[1,3,5,7,2,4,6], arrLen =7, indexA =5, indexB =10
    * After function call:   
      * print "Error: Index out of range." 
      * return=-1  
<p align="center">
<img width="610" alt="Screenshot 2022-10-02 at 00 15 32" src="https://user-images.githubusercontent.com/19373417/193418472-97904b61-545c-4224-8020-bda1a5b6e83a.png"></p>  

##Task 3
**Sort the array by rotation, and record the ks used in the rotations.**
void sortAndRecord(int arr[], int arrLen, int rotations[], int &rotationLen)

* Parameters:
  * arr : the array to be sorted.
  * arrLen : the length of arr. 
  * rotations : rotations to sort arr. 
  * rotationLen : length of rotations.  
  * arrLen : the length of arr. rotations : rotations to sort arr . rotationLen : length of rotations .
* Description:
  * Sort the elements of arr by rotations.
  * store the k values of the rotations performed in rotations .
  * store the count of rotations performed in rotationLen .
  * The answers are **NOT UNIQUE**. Any series of rotations that can sort the array are accepted, but the length of the rotation series should be **NO MORE THAN** MAX_ROTATIONS .
* Example
  * arr =[1,3,5,7,2,4,6], arrLen =7.
  * After function call:
    * arr =[1,2,3,4,5,6,7].
    * rotations =[3,6,0,5,0,4,0,3,0,2,0,1]. 
    * rotationLen =12.

##Task 4
**Transform the source array into the target by rotation, and record the ks used in the rotations.**
int transformAndRecord(int src[], int tgt[], int arrLen, int rotations[], int &rotationLen)。

* Parameters:
  * src : the source array to be transformed by rotations.
  * tgt : the target array that src should be transformed to. 
  * arrLen : the length of src and tgt.
  * rotations : actions to transform src to tgt. 
  * rotationLen : length of rotations .
* Description:
  * In this task you can assume src and tgt always have the same length. 
  * If arr and tgt have different elements:
    * return -1. 
  * Else:
    * transform arr to tgt by rotations.
    * store the k values of the rotations performed in rotations. 
    * store the count of rotations performed in rotationLen.
  * The answers are NOT UNIQUE. Any series of rotations that can sort the array are accepted, but the
length of the rotation series should be NO MORE THAN MAX_ROTATIONS. 

* Example:
1.  * src =[1,3,5,7,2,4,6], tgt =[1,5,3,7,2,4,6] arrLen =7. 
    * After function call:
      * src =[1,5,3,7,2,4,6]. 
      * rotations =[2,1]. 
      * rotationLen =2.
      * return=0.

2.  * src =[1,3,5,7,2,4,6], tgt =[1,5,3,7,2,4,7] arrLen =7.
    * After function call: 
      * return=-1.  
* Hint: the rotations are reversible, i.e., if a series of rotations can transform array A to B, then the reversed series of rotations can transform B to A.

Disclaimer:
This is a university project about how to solve a childhood boardgame the **Magic Array** problem, using C++ as required. The description of the problem comes from the Internet and school. However, the coding part is 100% original and I highly value the academic integrity.
