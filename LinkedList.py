class SinglyLinkedListNode:
    def __init__(self, node_data):
        self.data = node_data
        self.next = None


class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert_node(self, node_data):
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node


def insert_node_at_position(head, node_data, node_position):
    current_node = head
    for _ in range(node_position - 1):
        current_node = current_node.next
    new_node = SinglyLinkedListNode(node_data)
    new_node.next = current_node.next
    current_node.next = new_node
    return head


def find_merge_node(head1, head2):
    # detect x
    # a--->b--->c-->d-->e
    #                    \
    #                     x--->y--->z--->NULL
    #                    /
    #                   q
    a, b = head1, head2
    while a != b:
        a = a.next or head2
        b = b.next or head1
    return a


def has_cycle(head):
    # detect a cycle
    seen = {head}
    while head.next:
        head = head.next
        if head in seen:
            return True
        seen.add(head)
    return False


def print_singly_linked_list(node):
    while node:
        print(str(node.data), end=" ")
        node = node.next
    print("\n")


def reverse_list(head, prev=None):
    if not head:
        return prev
    curr, head.next = head.next, prev
    return reverse_list(curr, head)


if __name__ == '__main__':
    llist_count = int(input())
    llist = SinglyLinkedList()

    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)

    data = int(input())
    position = int(input())
    llist_head = insert_node_at_position(llist.head, data, position)
    print_singly_linked_list(llist_head)
