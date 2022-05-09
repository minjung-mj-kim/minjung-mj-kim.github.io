---
title:  "Linked List"
categories:
  - Cs-ds
tags:
  - Computer sciences
  - Data structure
  - Linked list
---

This post will briefly introduce about the linked list and its variations.

# Composition

- Head: Beginning of a list. 
- Node: Data + A pointer (link)
    - The pointer contains the address of the next node.
- Tail: End of a list. 

# Implementation

Following code shows a simple case. Implementation varies depending on the required actions.

```
class LinkedList:
    
    class Node:
        def __init__(self, data=None, link=None):
            self.data = data
            self.link = link

    def __init__(self,data):
        self.head = self.Node(data)
        
    def insert(self, data): # insert to head
        new_node = self.Node(data)
        if self.head:
            new_node.link = self.head
            self.head = new_node
            print('insert',new_node.data)
        else:
            self.head = new_node
            print('insert head',new_node.data)
        
    def remove(self): # remove head
        if self.head:
            print('remove',self.head.data)
            self.head = self.head.link
        else:
            print('no head. insert data first')

    def traversal(self):
        if self.head:
            node = self.head
            data_list = []
            while node:
                data_list.append(node.data)
                node = node.link
            print(data_list)
        else:
            print('Empty list')

# Test
ll = LinkedList(10)
ll.traversal()
ll.insert(9)
ll.traversal()
ll.insert(8)
ll.traversal()
ll.insert(7)
ll.traversal()
ll.remove()
ll.traversal()
ll.remove()
ll.traversal()
ll.remove()
ll.traversal()
ll.remove()
ll.traversal()
ll.insert(1)
ll.insert(2)
ll.insert(3)
ll.traversal()
```
Output:
```
[10]
insert 9
[9, 10]
insert 8
[8, 9, 10]
insert 7
[7, 8, 9, 10]
remove 7
[8, 9, 10]
remove 8
[9, 10]
remove 9
[10]
remove 10
Empty list
insert head 1
insert 2
insert 3
[3, 2, 1]

```


# Pros
- No memory waste from dynamic allocation.
- Insertion and deletion can be done anywhere and cost less (O(1) to insert in front of the head) than arrays (O(n)).

# Cons
- Complicated implementation.
- Fragmented memory allocation.
- Extra space than array for pointers.
- No direct access to an element in the middle.
- Traverse only one direction, from head to tail.


# Variations 

## Circular Linked List
- Same structure, except the last node points the first node.
- Traverse to the previous node is possible (on the next round).
- The head pointer is not necessary (head can be replaced by tail.link)

## Double Linked List
- Components:
    - Head: Beginning of a list. 
    - Node: Data + A pointer to the previous node + A pointer to the next node.
    - Tail: End of a list. 
- Can traverse bidirection.
    - Insertion and deletion of a node in front of a given node can be done fast.
- Requires more space for the previous pointers.
