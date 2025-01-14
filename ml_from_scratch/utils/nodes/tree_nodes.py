class Node:
    """
    Represents a general node for tree-based models.
    """
    def __init__(self, feature: int = None, threshold: float = None, left=None, right=None, *, value=None):
        """
        Initializes a node.

        Args:
            - feature (int): Index of the feature used for splitting (None for leaf nodes).
            - threshold (float): Threshold value for splitting (None for leaf nodes).
            - left (Node): Left child node.
            - right (Node): Right child node.
            - value (int | None): Class value for leaf nodes.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """
        Checks if the node is a leaf node.
        """
        return self.value is not None
