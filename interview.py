'''Given a number in an array form, Come up with an algorithm to push all the zeros to the end.
Expectation : O(n) solution
'''


def find_zeros(num):
	str_form = str(num)

	zero_position=-1
	for i in range(0, len(str_form)):
		if str_form[i] == "0":
			temp = str_form[i]
			str_form[i]=str_form[zero_position]
			str_form[zero_position]=temp
			zero_position-=1
	return int(str_form)



'''

How to check a tree is balanced? what if it is not BST?
'''

from math import *

def find_height(node,max_height=0, current_height=0, current_elements=1):
	right_height, left_height = current_height, current_height

	if node.right:
		right_height, number_of_right_elements = find_height(node.right, max_height=max_height, current_height = current_height+1, number_of_element=current_elements+1)
	else:
		return current_height
	if node.left:
		left_height, number_of_left_elements = find_height(node.left, max_height=max_height, current_height = current_height+1 ,number_of_elements = current_elements+1)
	else:
		return current_height

	max_height = max(left_height, right_height)

	number_of_elements = (number_of_right_elements-current_elements) + (number_of_left_elements-current_elements) + current_elements

	return max_height, number_of_elements

def is_balanced(node):
	max_height, number_of_elements = find_height(node)
	balanced_height = log(number_of_elements)
	if max_height >balanced_height:
		return False
	else:
		return True

from unittest import *
def verify_tree(node):

	if node.left:
		assert (node.left.data < node.data)
		verify_tree(node.left)

	if node.right:
		assert (node.right.data > node.data)
		verify_tree(node.right)


'''
Given two sorted arrays. Return the median of all elements in the two arrays.
'''

def median(array1, array2):
	index1=0
	index2=0
	sorted_array = []
	total_length = len(array1) + len(array2)
	med_index = total_length/2
	total_count=0
	while index1<len(array1) and index2 < len(array2):

		if array1[index1] < array2[index2]:
			sorted_array.append(array1[index1])
			index1+=1
		else:
			sorted_array.append(array2[index2])
			index2+=1
	return sorted_array[med_index-1]

arr1=[1,2,3]
arr2=[3,2,1]

print median(arr1, arr2)
'''
Given a number,find the next higher number using the same digits in the number. Eg- 15432, Soln- 21345.
use the first digit in the number.
find the next closest number to that digit
sort the digits after that number
'''

def find(number):
	min=None

	digits = str(number)
	digit_list=[]
	index = None
	for digit in digits:
		digit_list.append(int(digit))

	for i in range(1,len(number)):
		if number[i]>number[0]:
			if min is None:
				min = number[i]
				index = i

			else:
				if number[i] < min:
					min = number[i]
					index = i
	del digit_list[index]

	digit_list=sorted(digit_list)

	new_num = str(min)

	for digit in digit_list:
		new_num=new_num + str(digit)

	new_num = int(new_num)





'''
Given a Binary tree where nodes may have positive or negative value, store the sum of the left and right subtree in the nodes.
'''


def create_summation(node):

	if node is None:
		return 0

	right_sum = create_summation(node.right)
	left_sum =  create_summation(node.left)

	local_sum = left_sum + right_sum + node.data

	node.sum = local_sum

	return local_sum



class LinkedList():
	def __init__(self, data=5, next=None):
		self.data = data
		self.next=next

def concatenate(node):
	start_list = []
	start_list_data = []
	while(node.next != None):
		if node.data not in start_list_data:
			node.next = None
			start_list.append(node)
			start_list_data.append(node.data)
		else:
			for find_node in start_list:
				if node.data == find_node.data:
					next = find_node.next
					find_node.next = node
					node.next = next

	previous_node = None
	Root_Node=None

	start_list_data = sorted(start_list_data)

	sorted_nodes = []

	for i in range(0,len(start_list_data)):
		for j in range(0, len(start_list)):
			if start_list_data[i] == start_list[j]:

				sorted_nodes.append(start_list[j])

	for i in range(len(start_list)-1, -1, -1):

		if i ==0:
			Root_Node=start_list[i]


		if not previous_node:
			continue
		test = start_list[i].next
		while test!=None:
			test = start_list[i].next

		found_node = test
		found_node.next = previous_node



'''

For a given integer number, reverse the digits of the number.

convert to a string
reverse the string
'''

def reverse_integer(integer):
	type_cast = str(integer)
	new_str=""
	for i in range(len(type_cast)-1, -1, -1):
		new_str+=(type_cast[i])
	return new_str

#print reverse_integer(1234)






'''
Assume primitive Facebook. FB has Members.
class Member {
    String name;
    String email;
    List&lt;Member&gt; friends;
}
Question A:
Code printSocialGraph(Member m). Direct friends of m are Level 1 friends. Friends of friends are level 2 friends.....and so on
Print level 1 friends first. Then print level 2 friends....and so on

Enumerate test cases to verify the implementation is correct.

'''
class Member:
	def __init__(self, name, email, friends):
		self.name=name
		self.email=email
		self.friends = friends

def print_friend_lists(friend_lists):

	for each_list in friend_lists:
		print each_list

def breadth_first_search(self, member, level):
	friend_lists = []
	for l in level:
		new_list = []
		for friend in member.friends:

			already_found=False

			for friend_list in friend_lists:
				if friend in friend_list:
					already_found=True

				if not already_found:
					new_list.append(friend)

		friend_lists.append(new_list)

	print_friend_lists(friend_lists)



'''

k closest points to origin out of N on a map
You have a map which is a tree of nodes venturing from a starting point
use a hash as a stack to keep track of the closest nodes found
'''

class Node:
	def __init__(self, long, lat):
		#self.neighbor_nodes = self.create_neighbors(long, lat)
		self.longitude=long
		self.latitude=lat

	def create_neighbors(self, long, lat, levels):
		self.neighbor_nodes=[]
		for i in range(0, levels):
			curr_long = long + 1
			curr_lat = lat + 1
			self.neighbor_nodes.append(Node(curr_long, curr_lat))

class BinaryTree():
	def __init__(self, Node):
		self.tree_root=Node

	def add_element(self, node, distance):

		pass#while


def determine_distance(root, current_node):
	pass

def closest_points(root, current_root, hash):

	for current_node in current_root.neighbor_nodes:

		current_node.distance = determine_distance(root, current_node)

		hash.add_element(current_node, current_node.distance)

'''

long, lat = 1,1

new_node = Node(long, lat)

new_node.create_neighbors(long, lat, 4)

tree=BinaryTree(new_node)

closest_points(tree, tree, hash)
'''


'''
Write a function that takes two lists of strings and return a list of Strings with all of the intersections of the strings ex:

List1 = {"a","a","a", "b", "d"}
List2 = {"a", "a", "c", "d"}
expectedReturn={"a","a","d"}

Also he asked what tests cases I would use to validate the function also he wanted to know the run time analysis of the function

put each string in dictionary 1 with a counter of # occurences
iterate the second list, decrementing the counter for each found string in dictionary_1


'''
List1 = ["a","a","a", "b", "d"]
List2 = ["a", "a", "c", "d"]


def find_matches(List1, List2):

	matches=list()
	dict_1={}
	for s in List1:
		if s in dict_1:
			dict_1[s]= dict_1[s]+1
			print "incrementing " + s
		else:
			dict_1[s] =1
	print dict_1
	for s2 in List2:
		if s2 in dict_1:
			print "found"
			if dict_1[s2] >0:
				matches.append(s2)
			dict_1[s2]-=1

	print dict_1
	return matches
#print find_matches(List1, List2)

'''
Given a BST and a value x. Find two nodes in the tree whose sum is equal x. Additional space: O(height of the tree). It is not allowed to modify the tree


Find the furthest left node, and the furthest right node that is less than the desired value

'''
'''

50
40 80
10 45 70 90
'''


'''
Given three arrays A,B,C containing unsorted numbers. Find three numbers a, b, c from each of array A, B, C such that |a-b|, |b-c| and |c-a| are minimum
Please provide as efficient code as you can.
'''

def find_minimum_divergence(A, B, C):
	A_pointer, B_pointer, C_pointer=0,0,0

	A, B, C = sorted(A), sorted(B), sorted(C)

	best_diff=None


	while(A_pointer<len(A)-1 and B_pointer<len(B)-1 and C_pointer<len(C)-1):
		current_diff1 = abs(A[A_pointer] - B[B_pointer])
		current_diff2 = abs(B[B_pointer] - C[C_pointer])
		current_diff3 = abs(C[C_pointer] - A[A_pointer])

		total_diff = current_diff2 + current_diff1 + current_diff3

		if best_diff is None:

			best_diff = total_diff

			best_A = A_pointer
			best_B = B_pointer
			best_C = C_pointer

		elif total_diff < best_diff:


			best_diff = total_diff

			best_A = A_pointer
			best_B = B_pointer
			best_C = C_pointer

		if A[A_pointer] < B[B_pointer]:
			if A[A_pointer] < C[C_pointer]:
				if len(A)-1 > A_pointer:
					A_pointer +=1
				elif len(C)-1 > C_pointer:
					C_pointer +=1
				else:
					B_pointer +=1
			else:
				if len(C)-1 > C_pointer:
					C_pointer +=1
				elif len(A)-1 > A_pointer:
					A_pointer +=1
				else:
					B_pointer +=1
		else:
			if B[B_pointer] < C[C_pointer]:
				if len(B)-1 > B_pointer:
					B_pointer +=1
				elif len(C)-1 > C_pointer:
					C_pointer +=1
				else:
					A_pointer +=1

			else:

				if len(C)-1 > C_pointer:
					C_pointer +=1
				elif len(B)-1 > B_pointer:
					B_pointer +=1
				else:
					A_pointer +=1
	return (best_A, best_B, best_C)



array_a = [1,5,7,8,2,-2,4]

array_b= [5,77,4,3,9,66,0,-1]

array_c = [6,5,22,44,11,3]

(best_A, best_B, best_C)= find_minimum_divergence(array_a, array_b, array_c)
#print best_A, best_B, best_C


'''divide and conquer largest subsequence sum
'''


'''
def largest_sum(array):
	if len(array)>2:
		pivot = len(array)/2
		left_max = largest_sum(array[:pivot])
		right_max = largest_sum(array[pivot:])
		current_max = max([left_max, right_max, (left_max + right_max)])

	elif len(array) <= 1:
		return array[0]
	elif len(array) ==2:
		return max([array[0], array[1], (array[0] +array[1])])

arr=[1,0,-5,6,8,-18,-29,1,5,0,-3,3,4]
result = largest_sum(arr)
#print result

'''


'''
How will you detect a loop in linked list.

iterate the linked list with two pointers, one which iterates one link at a time, the
other increments two at a time.
check for the time when the two pointers cr0ss
they will become equal at the midpoint the second time they cross
'''



'''
for each possible pivot point, compare alternating incrementing the right then left sides
if the result is not a palindrome after incrementing both, continue, otherwise add to palindrome list
'''



'''
Merge the given 2 input sorted arrays of numbers into one . The merged array stays sorted .
'''


def merge(array1, array2):
	len1 = len(array1)
	len2 = len(array2)

	i,j=0, 0
	new_array=[]
	while i<len1 or j<len2:
		if array1[i] < array2[j]:
			new_array.append(array1[i])
			i+=1
		elif array1[i] > array2[j]:
			new_array.append(array2[j])
			j+=1
		else:
			new_array.append(array1[i])
			i+=1
			new_array.append(array2[j])
			j+=1
		if i==len1:
			while j < len2:
				new_array.append(array2[j])
				j+=1
		elif j==len2:
			while i<len1:
				new_array.append(array1[i])
				i+=1
		print new_array
	return new_array

arr1 = [1,5,8,99,456,4566]
arr2 = [2,7,3,8,3,45,32453245]
#test = merge(arr1, arr2)
'''
Binary search inorder traversal asked by Amazon
class Node
{
int data;
Node *right.*left,*random
}

Tree should be in-order traversal and random node should keep the in-order transversal path.

traverse to the leftmost nodes
while also recursively calling the right side, which should execute after the
'''
def radix(array):
	radix = 10
	mult = 1
	dan = True

	while dan is True:
		dan = False
		print "iterating"

		buckets = [list() for x in range(radix)]

		for element in array:
			bucket_mult = element/mult
			digit = bucket_mult%radix
			if bucket_mult >0 and dan is False:
				dan=True
			buckets[digit].append(element)
		array = []
		for bucket in buckets:

			array += bucket


		mult *=10

	return array





def quick_sort(array):

	if len(array) > 1:
		pivot = array[0]
		right_swap = pivot+1
		pivot_index=0
		less = []
		equal=[]
		greater=[]
		i=0
		while i < len(array):
			if array[i]>pivot:
				greater.append(array[i])
			elif array[i]<pivot:
				less.append(array[i])
			else:
				equal.append(array[i])
			i+=1
		return quick_sort(less) + equal + quick_sort(greater)
	else:
		return array

'''
array = [2,5,222,3,4,83,1,475,23,523,4,45,456,45,456]

test = radix(array)

print test
'''
def fibonacci_recursive(max, i=1, stack = [0]):
	stack.append(i)
	if len(stack)<max:
		i+=stack[-2]
		stack = fibonacci_recursive(max,i,stack)
	return stack


def fib(max):
	seq = [0,1]
	for i in range(2, max):
		print i
		seq.append(seq[-1] + seq[-2])

	return seq

'''
stack=fib(8)
print stack
'''
def ordered_search(stack, node, level = 0, deepest_level = 0):
	level += 1
	if type(stack) != list:
		print "wrong stack type"
		return None

	if type(node) != Node:
		print "wrong node type"
		return None
	else:
		if node.left is not None:
			stack = ordered_search(stack, node.left, level=level)
		stack.append(level)
		if node.right is not None:
			stack = ordered_search(stack, node.right, level=level)
		return stack



'''
	Wap to find kth largest element in a binary search tree
'''


class Node:
	def __init__(self, left, right):
		self.right=right
		self.left=left











'''
	Wap to find kth largest element in a binary search tree
'''


def ordered_search(node,rank_returned, rank=1):

	if rank == rank_returned:
		return (rank, node)

	if node.left is not None:
		current_rank = ordered_search(node.left, rank=rank+1)
		if type(current_rank) is not int:
			return current_rank
		elif current_rank == rank_returned:
			return node

	if node.right is not None:
		current_rank = ordered_search(node.right, rank=rank+1)
		if type(current_rank) is not int:
			return current_rank
		elif current_rank == rank_returned:
			return node



class Node:
	def __init__(self):
		data=int()
		right=Node()
		left=Node()
		random = Node()

def palindrome(test_str):

	if len(test_str) %2 !=0:
		return False
	else:
		i, j=0
		while i<=len(test_str)/2:
			if test_str[i] != test_str[j]:
				return false
		return True



def ordered_search(stack, node):
	if type(stack) != list:
		print "wrong stack type"
		return None

	if type(node) != Node:
		print "wrong node type"
		return None
	else:
		if node.left is not None:
			item = ordered_search(node.left)
		print node.data
		if node.right is not None:
			item = ordered_search(node.right)
		else:
			return node.data

'''
use a dictionary to store the values in the first array
iterate second, see if key exists in dictionary, change value to "duplicate"
'''

'''
iterate third array, if any values are not "duplicate", add to unique array
'''


def tree_depth(root,current_level=1, level_stack=[]):
	if root.left is None:
		level_stack.append(current_level)

	else:
		level_stack = tree_depth(root.left, current_level+1,level_stack)
	if root.right is None:
		level_stack.append(current_level)
	else:
		level_stack = tree_depth(root.left, current_level+1,level_stack)

	return level_stack

def recurse(min, max, i):
	sum=0
	if i <= max:
		sum += recurse(min, max, i+1) + i
		print sum
		return sum
	else:
		return 0

'''
Given a list of integers of size n, write a method to write all permutations the list; do this in O(1) space
Hint: No recursion.

could create a sting of all the integers,
create and alogithm to find all nombinations of strs of length 1-n where n=string size
'''
def permutations(head, tail=''):
	if len(head) == 0: print tail
	else:
		for i in range(len(head)):
			print "calling 	permutations {} + {}, {}".format(head[0:i], head[i+1:], tail+head[i])
			permutations(head[0:i] + head[i+1:], tail+head[i])

#permutations('abcd')

def combinations(head, stored=[]):
	total_str = ""
	if type(head) != list:
		print "wrong type given"
		return

	if len(head) == 0:
		print stored
	for i in range(0, len(head)):

		combinations(head[0:i]+ head[i+1:], stored =(stored + list(head[i])))

#combinations(["dan", "wants", "food"])


def perm(a,k=0):
   if(k==len(a)):
      print a
   else:
      for i in xrange(k,len(a)):
         a[k],a[i] = a[i],a[k]
         perm(a, k+1)
         a[k],a[i] = a[i],a[k]

#perm([1,2,3])

'''Binary search inorder traversal asked by Amazon
struct Node
{
int data;
Node *right.*left,*random
}

Tree should be in-order traversal and random node should keep the in-order transversal path.'''

#test = recurse(1,4,1)
print "dan"
#print test


'''
Given a base longitude base_long and base latitude base_lat
I would create a binary tree based on the pythagorean theorem
distance between sqrt(base_long - long)^2 + (base_lat - lat)^2
and recuresively print the leftmost node value
'''


