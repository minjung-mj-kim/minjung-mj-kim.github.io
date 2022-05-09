---
title:  "Stack, Queue, and Deque"
categories:
  - Cs-ds
tags:
  - Computer sciences
  - Data structure
  - Stack
  - Queue
  - Deque
---

Stack, Queue, and Deque are commonly used linear (ordered) data strucrues.
They vary depending on the position of insertion and deletion operation.

# Stack

- Insertion and deletion can be done only at the top.
- LIFO: Last-In First-Out
- Basic operations
    - Push: insert/stack the new element on the top
    - Pop: remove the last (top) element
- Stack overflow: push an item when the stack is full.

## Implementation

Both array and linked list can be used.
Here, linked list based implementation is shown.
Both push and pop are done in the head here.
About the memory location, all you care is for the top element.

```
class Stack:
    class Node:
        def __init__(self, data=None, link=None):
            self.data = data
            self.link = link
    
    def __init__(self, max_size=100):
        self.top = None
        self.size = 0
        self.max_size = max_size

    def push(self, data):
        if self.max_size>self.size:
            new_node = self.Node(data)
            new_node.link = self.top
            self.top = new_node
            self.size += 1
            print('now top is:',self.top.data)
        else:
            print('stack overflow')

    def pop(self):
        if self.size>0:
            node_to_remove = self.top
            self.top = node_to_remove.link
            self.size -=1
            try:
                print('pop',node_to_remove.data,
                      ' now top is:',self.top.data)
            except:
                print('that was the last item')
            return node_to_remove.data
        else:
            print('Empty stack')
  
  
stack = Stack(3)
stack.push("1")
stack.push("2")
stack.push("3")
stack.push("4")
stack.pop()
stack.pop()
stack.pop()
stack.pop()
```
Output:
```
now top is: 1
now top is: 2
now top is: 3
stack overflow
pop 3  now top is: 2
pop 2  now top is: 1
that was the last item
stack is empty
```


## Application 
- Function call
- Recursive function
- Parenthesis matching
- Full subway


# Queue
- Insertion (deletion) can be done only at the rear (front).
- FIFO: First-In First-Out
- Basic operations
    - Enqueue: insert the new element at the end of a queue
    - Dequeue: remove the oldest element
    - Enqueue and dequeue occur at the opposit sides
- Stack overflow: push an item when the stack is full.

## Implementation

In case of array-based implementation, each dequeue operation increases the number of empty array elements from the front, 
while each enqueue operation increases the largest array index. 
Such case can be efficiently handled by using the circular array.
Linked list can be another implementation choice, which is shown in the following code block.

```
class Queue:
    class Node:
        def __init__(self, data=None, link=None):
            self.data = data
            self.link = link
    
    def __init__(self, max_size=100):
        self.head = None
        self.tail = None
        self.size = 0
        self.max_size = max_size

    def enqueue(self, data):
        if self.max_size>self.size:
            new_node = self.Node(data)
            if self.size==0:
                self.head = new_node
                self.tail = new_node
            else:
                self.tail.link = new_node
                self.tail = new_node
            self.size+=1
        else:
            print('This queue is full')

    def dequeue(self):
        if self.size>0:
            node_to_remove = self.head
            if self.size==1:
                self.head=None
                self.tail=None
            else:
                self.head = self.head.link
            self.size-=1
            return node_to_remove.data
        else:
            print('This queue is empty')
    
    def traversal(self):
        if self.head:
            node = self.head
            data_list = []
            while node:
                data_list.append(node.data)
                node = node.link
            print(data_list)
        else:
            print('This queue is empty')

q = Queue(3)
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.traversal()

q.dequeue()
q.traversal()
q.dequeue()
q.dequeue()
q.dequeue()
```
Output:
```
[1, 2, 3]
[2, 3]
This queue is empty
```


## Application 
- OS job scheduling
- Public bathroom

# Deque
Deque is double ended queue, like Stack+Queue.
Enqueue and Dequeue can be done in both ends.
Python provides a library of deque.
```
from collections import deque 
```