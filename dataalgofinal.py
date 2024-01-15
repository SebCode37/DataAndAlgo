import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import isinf
[ ]: #Import of the cities into Python
cities = pd.read_csv('cities.csv')
# avoid run time errors - bug fix for display formats
pd.set_option('display.float_format', lambda x:'%f'%x)
cities = cities.apply(pd.to_numeric, errors='coerce')
cities.head()
[ ]: cities.CityId
[ ]: #scatter plot printing locations of cities from 0-10 (for demonstration␣
,→purposes)
%matplotlib notebook
fig = plt.figure(figsize=(10,10))
plt.scatter(cities[cities['CityId']<=10].X , cities[cities['CityId']<=10].Y, s=␣
,→200, color = 'red')
0.0.1 Initial Algorithm and Data Structure(List Data Structure)
The data structure used in this algorithm is a list.
This function does not have a linear run time as it contains a while loop nested in a for loop
At worst, it will run through n elements n times O(n^2) List comprehension runs n times with 2
operations (assignment, sequence access) 2 primitive operations are run (assignment) For loop runs
n times If statement runs 1 time with 1 primitive operations inside it A while loop runs n times
with 2 operations
[ ]: #Determine which cities are prime numbers
def sieve_of_eratosthenes(n):
primes = [True for i in range(n+1)] # Start assuming all numbers are primes
primes[0] = False # 0 is not a prime
primes[1] = False # 1 is not a prime
for i in range(2,int(np.sqrt(n)) + 1):
1

if primes[i]:
k = 2
while i*k <= n:
primes[i*k] = False
k += 1
return(primes)

[ ]: cities['is_prime'] = sieve_of_eratosthenes(max(cities.CityId))
[ ]: prime_cities = sieve_of_eratosthenes(max(cities.CityId))
[ ]: #Visualising the map with the prime cities highlighted.
fig = plt.figure(figsize=(10,10))
plt.scatter(cities[cities['CityId']==0].X , cities[cities['CityId']==0].Y, s=␣
,→200, color = 'red')
plt.scatter(cities[cities['is_prime']==True].X ,␣
,→cities[cities['is_prime']==True].Y, s= 0.8, color = 'purple')
plt.scatter(cities[cities['is_prime']==False].X ,␣
,→cities[cities['is_prime']==False].Y, s= 0.1)
plt.show()
[ ]: import time
start_time = time.time()
def pair_distance(x,y):
x1 = (cities.X[x] - cities.X[y]) ** 2
x2 = (cities.Y[x] - cities.Y[y]) ** 2
return np.sqrt(x1 + x2)
end_time = time.time()
dumbest_elapsed_time = end_time - start_time
[ ]: print("Total elapsed time of dumbest path algorithm: ", dumbest_elapsed_time)
[ ]: def total_distance(path):

distance = [pair_distance(path[x], path[x+1]) + 0.1 *␣
,→pair_distance(path[x], path[x+1])
if (x+1)%10 == 0 and cities.is_prime[path[x]] == False else␣
,→pair_distance(path[x], path[x+1]) for x in range(len(path)-1)]
return np.sum(distance)

[ ]: dumbest_path = cities['CityId'].values
#add North Pole add the end of trip
dumbest_path = np.append(dumbest_path,0)
[ ]: print('Total distance with the paired city path is '+ "{:,}".
,→format(total_distance(dumbest_path)))

2

0.0.2 Quick Sort Algorithm - Divide and Conquer Strategy
[ ]: City_X=[]
for x in range(max(cities.CityId)+1):
City_X.append(cities['X'][x])
City_Y=[]
for x in range(max(cities.CityId)+1):
City_Y.append(cities['Y'][x])

[ ]: path=[]
for x in range(1,max(cities.CityId)+1):
path.append(x)
[ ]: def partition(arr,low,high):

i = ( low-1 ) # index of smaller element
pivot = arr[high] # pivot
for j in range(low , high):
# If current element is smaller than or
# equal to pivot
if City_X[arr[j]] <= City_X[pivot]:
# increment index of smaller element
i = i+1
arr[i],arr[j] = arr[j],arr[i]
arr[i+1],arr[high] = arr[high],arr[i+1]
return ( i+1 )
[ ]: start_time = time.time()
def quickSort(arr,low,high):
if low < high:
# pi is partitioning index, arr[p] is now
# at right place
pi = partition(arr,low,high)
# Separately sort elements before
# partition and after partition
quickSort(arr, low, pi-1)
quickSort(arr, pi+1, high)
end_time = time.time()
quicksort_elapsed_time = end_time - start_time
[ ]: print("Total elapsed time of quicksort algorithm: ", quicksort_elapsed_time)

3

[ ]: quicksort_path=[]
for x in range(1,max(cities.CityId)+1):
quicksort_path.append(x)

[ ]: quickSort(quicksort_path,0,len(quicksort_path)-1)
[ ]: quicksorted_path=[0]
for each in range(len(quicksort_path)):
quicksorted_path.append(quicksort_path[each])
quicksorted_path.append(0)
[ ]: print('Total distance with the quick sorted cities based on X path is '+ "{:,}".
,→format(total_distance(quicksorted_path)))
0.0.3 Binary Search Tree
[ ]: from random import randint
tree_path = []
class node:
def __init__(self, value = None):
self.value = value
self.left = None
self.right = None
class binary_search_tree:
def __init__(self):
self.root = None
def insert(self, value):
if self.root == None:
self.root = node(value)
else:
self._insert(value, self.root)

def _insert(self, value, cur_node):
if City_Y[value] <= City_Y[cur_node.value]:
if cur_node.left == None:
cur_node.left = node(value)
else:
self._insert(value, cur_node.left)
elif City_X[value] >= City_X[cur_node.value]:
if cur_node.right == None:
cur_node.right = node(value)
else:
self._insert(value, cur_node.right)
else:

4

tree_path.append(path[cur_node.value])

def print_tree(self):
if self.root != None:
self._print_tree(self.root)

def _print_tree(self, cur_node):
if cur_node != None:
self._print_tree(cur_node.left)
tree_path.append(path[cur_node.value])
self._print_tree(cur_node.right)

def fill_tree(tree):
for x in range(len(path)-1):
cur_elem = path[x]
tree.insert(cur_elem)
return tree
tree = binary_search_tree()
tree = fill_tree(tree)
tree.print_tree()
print('Total distance with the Binary Tree sorting city based on X path is '+␣
,→"{:,}".format(total_distance(tree_path)))

[ ]: def submission():

dict = {'Path': tree_path}
df = pd.DataFrame(dict)
#write data from dataframe to csv file
df.to_csv('Final_Submission.csv', index=False)

[ ]: submission()
