#include <iostream>
#include <cstring>
#include "cleaning_robot.h"
using namespace std;

// Direction arrays: Up, Right, Down, Left
const int dx[4] = {0, 1, 0, -1};
const int dy[4] = {-1, 0, 1, 0};
const char dirChar[4] = {'U', 'R', 'D', 'L'};

// Helper: check if (x, y) is in bounds
inline bool inBounds(int x, int y) {
    return x >= 0 && x < MAP_WIDTH && y >= 0 && y < MAP_HEIGHT;
}

// Please do all your work in this file. You just need to submit this file.
//Task 1
int findMaximumPlace(int robot_x, int robot_y, int robot_energy, int robot_full_energy, 
   char result_map[MAP_HEIGHT][MAP_WIDTH], char temp_map[MAP_HEIGHT][MAP_WIDTH])
{
   if (!inBounds(robot_x, robot_y) || robot_energy < 0)
       return 0;
   // Blocked or already visited
   if (result_map[robot_y][robot_x] == BLOCKED || result_map[robot_y][robot_x] == VISITED)
       return 0;
   // Mark as visited
   char prev = result_map[robot_y][robot_x];
   result_map[robot_y][robot_x] = VISITED;
   int count = 1;
   // Recharge if on charger
   if (prev == CHARGER)
       robot_energy = robot_full_energy;
   // Explore 4 directions
   for (int d = 0; d < 4; ++d) {
       int nx = robot_x + dx[d];
       int ny = robot_y + dy[d];
       if (inBounds(nx, ny) && result_map[ny][nx] != BLOCKED && result_map[ny][nx] != VISITED) {
           count += findMaximumPlace(nx, ny, robot_energy - 1, robot_full_energy, result_map, temp_map);
       }
   }
   return count;
}

//Task 2
int minValue(int a, int b, int c, int d){
   return min(min(a,b), min(c,d));
}

int findShortestDistance(int robot_x, int robot_y, int target_x, int target_y, int robot_energy, 
   int robot_full_energy, const char map[MAP_HEIGHT][MAP_WIDTH], char temp_map[MAP_HEIGHT][MAP_WIDTH])
{   
   if (!inBounds(robot_x, robot_y) || robot_energy < 0)
       return PA2_MAX_PATH;
   if (map[robot_y][robot_x] == BLOCKED)
       return PA2_MAX_PATH;
   if (temp_map[robot_y][robot_x] == 'G') // visited in this path
       return PA2_MAX_PATH;
   if (robot_x == target_x && robot_y == target_y)
       return 1;
   // Mark as visited
   temp_map[robot_y][robot_x] = 'G';
   int minDist = PA2_MAX_PATH;
   int next_energy = (map[robot_y][robot_x] == CHARGER) ? robot_full_energy : robot_energy;
   for (int d = 0; d < 4; ++d) {
       int nx = robot_x + dx[d];
       int ny = robot_y + dy[d];
       if (inBounds(nx, ny)) {
           int dist = findShortestDistance(nx, ny, target_x, target_y, next_energy - 1, robot_full_energy, map, temp_map);
           if (dist < minDist)
               minDist = dist;
       }
   }
   temp_map[robot_y][robot_x] = 0; // unmark for other paths
   if (minDist == PA2_MAX_PATH)
       return PA2_MAX_PATH;
   return minDist + 1;
}
  



/*
int findShortestDistance(int robot_x, int robot_y, int target_x, int target_y, int robot_energy, 
   int robot_full_energy, const char map[MAP_HEIGHT][MAP_WIDTH], char temp_map[MAP_HEIGHT][MAP_WIDTH])
{  
   if ((robot_y == target_y) && (robot_x == target_x)){temp_map[0][0] = 1;}
   if ( robot_y >= 0 && robot_y <= MAP_HEIGHT && robot_x >= 0 && robot_x <= MAP_WIDTH && robot_energy > 0)
   {
      cout << "Current: " << robot_x << robot_y << endl;
      if (map[robot_y][robot_x] == 'C'){robot_energy = robot_full_energy+1;}     
      //up right down left
      if ((map[robot_y-1][robot_x] != 'B')&&(robot_y-1 >= 0)){
         cout << "U" << robot_x << robot_y-1 << endl;
         findShortestDistance(robot_x, robot_y-1, target_x, target_y, robot_energy-1, 
         robot_full_energy, map, temp_map);}
         

      if ((map[robot_y][robot_x+1] != 'B')&&(robot_x+1 <= MAP_WIDTH)){
         cout << "R" << robot_x+1 << robot_y << endl; 
         findShortestDistance(robot_x+1, robot_y, target_x, target_y, robot_energy-1, 
         robot_full_energy, map, temp_map);}
         

      if ((map[robot_y+1][robot_x] != 'B')&&(robot_y+1 <= MAP_HEIGHT)){
         cout << "D" << robot_x << robot_y+1 << endl;
         findShortestDistance(robot_x, robot_y+1, target_x, target_y, robot_energy-1, 
         robot_full_energy, map, temp_map);}
         

      if ((map[robot_y][robot_x-1] != 'B')&&(robot_x-1 >= 0)){
         cout << "L" << robot_x-1 << robot_y << endl;
         findShortestDistance(robot_x-1, robot_y, target_x, target_y , robot_energy-1, 
         robot_full_energy, map, temp_map);}
         
      
      
         
   }
   if ((robot_y == target_y) && (robot_x == target_x)){
      return static_cast<int>(temp_map[0][0]);}
}
*/

//Task 3
int findPathSequence(int robot_x, int robot_y, int target_x, int target_y, int robot_energy, int robot_full_energy, 
   char result_sequence[], const char map[MAP_HEIGHT][MAP_WIDTH], char temp_map[MAP_HEIGHT][MAP_WIDTH])
{
   if (!inBounds(robot_x, robot_y) || robot_energy < 0)
       return PA2_MAX_PATH;
   if (map[robot_y][robot_x] == BLOCKED)
       return PA2_MAX_PATH;
   if (temp_map[robot_y][robot_x] == 'G')
       return PA2_MAX_PATH;
   if (robot_x == target_x && robot_y == target_y) {
       result_sequence[0] = 'T';
       result_sequence[1] = '\0';
       return 1;
   }
   temp_map[robot_y][robot_x] = 'G';
   int minDist = PA2_MAX_PATH;
   int bestDir = -1;
   char bestSeq[MAX_STRING_SIZE];
   int next_energy = (map[robot_y][robot_x] == CHARGER) ? robot_full_energy : robot_energy;
   for (int d = 0; d < 4; ++d) {
       int nx = robot_x + dx[d];
       int ny = robot_y + dy[d];
       if (inBounds(nx, ny)) {
           char subSeq[MAX_STRING_SIZE];
           int dist = findPathSequence(nx, ny, target_x, target_y, next_energy - 1, robot_full_energy, subSeq, map, temp_map);
           if (dist < minDist) {
               minDist = dist;
               bestDir = d;
               strcpy(bestSeq, subSeq);
           }
       }
   }
   temp_map[robot_y][robot_x] = 0;
   if (minDist == PA2_MAX_PATH)
       return PA2_MAX_PATH;
   // Build result sequence: direction + subpath
   result_sequence[0] = dirChar[bestDir];
   strcpy(result_sequence + 1, bestSeq);
   return minDist + 1;
}

//Task 4
int findFarthestPossibleCharger(int robot_x, int robot_y, int robot_original_x, int robot_original_y, 
   int &target_x, int &target_y, int robot_energy, int robot_full_energy, const char map[MAP_HEIGHT][MAP_WIDTH], 
   char temp_map[MAP_HEIGHT][MAP_WIDTH])
{
   int maxDist = -1;
   int best_x = -1, best_y = -1;
   // For each cell, if it's a charger and reachable, check distance
   for (int y = 0; y < MAP_HEIGHT; ++y) {
       for (int x = 0; x < MAP_WIDTH; ++x) {
           if (map[y][x] == CHARGER) {
               char temp_map2[MAP_HEIGHT][MAP_WIDTH];
               memset(temp_map2, 0, sizeof(temp_map2));
               int dist = findShortestDistance(robot_x, robot_y, x, y, robot_energy, robot_full_energy, map, temp_map2);
               if (dist < PA2_MAX_PATH && dist > maxDist) {
                   maxDist = dist;
                   best_x = x;
                   best_y = y;
               }
           }
       }
   }
   if (maxDist == -1)
       return -1;
   target_x = best_x;
   target_y = best_y;
   return maxDist;
}