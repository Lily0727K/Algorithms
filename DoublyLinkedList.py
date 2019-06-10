class DoublyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert_node(self, node_data):
        node = DoublyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
            node.prev = self.tail
        self.tail = node


def print_doubly_linked_list(node):
    while node:
        print(str(node.data), end=" ")
        node = node.next
    print("\n")


def sorted_insert(head, node_data):
    new_node = DoublyLinkedListNode(node_data)
    if node_data < head.data:
        new_node.next = head
        head.prev = new_node
        return new_node

    current_node = head
    while current_node.next:
        current_node = current_node.next
        if node_data < current_node.data:
            current_node.prev.next = new_node
            new_node.prev = current_node.prev
            new_node.next = current_node
            current_node.prev = new_node
            return head

    new_node.prev = current_node
    current_node.next = new_node
    return head


def reverse(head):
    head.prev, head.next = head.next, head.prev
    if head.prev:
        return reverse(head.prev)
    else:
        return head


if __name__ == '__main__':
    llist_count = int(input())
    llist = DoublyLinkedList()
    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)

    insert_data = int(input())
    llist1 = sorted_insert(llist.head, insert_data)
    print_doubly_linked_list(llist1)

    llist2 = reverse(llist.head)
    print_doubly_linked_list(llist2)
