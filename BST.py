class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def sorted_array_to_binary_search_tree(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_binary_search_tree(nums[:mid])
    root.right = sorted_array_to_binary_search_tree(nums[mid + 1:])
    return root


