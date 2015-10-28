from abc import ABCMeta, abstractmethod


class Tree:
    __metaclass__ = ABCMeta

    class Position:
        __metaclass__ = ABCMeta

        @abstractmethod
        def element(self):
            pass

        @abstractmethod
        def __eq__(self, other):
            pass

        def __ne__(self, other):
            return not self.__eq__(other)

    @abstractmethod
    def root(self):
        raise NotImplementedError('must be implemented by subclass')

    def leaf(self):
        raise NotImplementedError('must be implemented by subclass')

    @abstractmethod
    def children(self, p):
        raise NotImplementedError('must be implemented by subclass')

    @abstractmethod
    def parent(self, p):
        raise NotImplementedError('must be implemented by subclass')

    @abstractmethod
    def num_children(self, p):
        raise NotImplementedError('must be implemented by subclass')

    @abstractmethod
    def __len__(self):
        raise NotImplementedError('must be implemented by subclass')

    def is_empty(self):
        return len(self) == 0

    def is_root(self, p):
        return self.root() == p

    def is_leaf(self, p):
        return self.num_children(p) == 0

    def positions(self):
        return self.preorder()

    def depth(self, p):
        if self.is_root(p):
            return 0
        else:
            return 1 + self.depth(self.parent(p))

    def _height1(self):
        return max(self.depth(p) for p in self.positions() if self.is_leaf(p))

    def _height2(self, p):
        if self.is_leaf(p):
            return 0
        else:
            return 1 + max(self._height2(c) for c in self.children(p))

    def height(self, p=None):
        if p is None:
            return self.root()
        return self._height2(p)

    def __iter__(self):
        for p in self.positions():
            yield p.element()

    def preorder(self):
        if not self.is_empty():
            yield self.preorder()

    def _subtree_preorder(self, p):
        yield p
        for c in self.children(p):
            for other in self._subtree_preorder(c):
                yield other

    def postorder(self):
        if not self.is_empty():
            yield self.postorder()

    def _subtree_postorder(self, p):
        for c in self.children(p):
            for other in self._subtree_postorder(c):
                yield other
        yield p


class BinaryTree(Tree):
    @abstractmethod
    def left(self, p):
        raise NotImplementedError('must be implemented by subclass')

    @abstractmethod
    def right(self, p):
        raise NotImplementedError('must be implemented by subclass')

    def sibling(self, p):
        parent = self.parent(p)
        if parent is None:
            return None
        else:
            if p == self.left(parent):
                return self.right(parent)
            else:
                return self.left(parent)

    def children(self, p):
        if self.left(p) is not None:
            yield self.left(p)
        if self.right(p) is not None:
            yield self.right(p)

    def inorder(self):
        if not self.is_empty():
            yield self._subtree_inorder(self.root())

    def _subtree_inorder(self, p):
        if self.left(p) is not None:
            for other in self._subtree_inorder(self.left(p)):
                yield other
        yield p
        if self.right(p) is not None:
            for other in self._subtree_inorder(self.right(p)):
                yield other

    def positions(self):
        return self.inorder()

    def num_children(self, p):
        count = 0
        if self.left(p) is not None:
            count += 1
        if self.right(p) is not None:
            count += 1
        return count


class LinkedBinaryTree(BinaryTree):
    # --------------------- nested class _Node and Position --------------- #
    class _Node:
        def __init__(self, element, parent=None, left=None, right=None):
            self._element = element
            self._parent = parent
            self._left = left
            self._right = right

    class Position(BinaryTree.Position):
        def __init__(self, container, node):
            self._container = container
            self._node = node

        def element(self):
            return self._node._element

        def __eq__(self, other):
            return type(other) == type(self) and other._node is self._node

    # ------------------------------ accessor method ----------------------- #
    def _validate(self, p):
        if not isinstance(p, self.Position):
            raise TypeError('p must be an instance of Position')
        if p._container is not self:
            raise ValueError('p must belong to this container')
        if p._node._parent == p._node:
            raise ValueError('p is no longer valid')
        else:
            return p._node

    def _make_position(self, node):
        return self.Position(self, node) if node is not None else None

    def __init__(self):
        self._root = None
        self._size = 0

    def root(self):
        """:return position of root"""
        return self._make_position(self._root)

    def __len__(self):
        return self._size

    def left(self, p):
        node = self._validate(p)
        return self._make_position(node._left)

    def right(self, p):
        node = self._validate(p)
        return self._make_position(node._right)

    def parent(self, p):
        node = self._validate(p)
        if node is self.root()._node:
            return None
        else:
            return self._make_position(node._parent)

    def children(self, p):
        node = self._validate(p)
        if node._left is not None:
            yield self._make_position(node._left)
        if node._right is not None:
            yield self._make_position(node._right)

    def is_root(self, p):
        node = self._validate(p)
        return node._parent is None

    def is_leaf(self, p):
        node = self._validate(p)
        return node._left is None and node._right is None

    def num_children(self, p):
        node = self._validate(p)
        count = 0
        if node._left is not None:
            count += 1
        if node._right is not None:
            count += 1
        return count

    # ------------------------- mutable method ----------------------- #
    def _add_root(self, e):
        if self._root is not None:
            raise ValueError('root already exists')
        else:
            self._root = self._Node(e)
            self._size += 1
        return self._make_position(self._root)

    def _add_left(self, p, e):
        node = self._validate(p)
        if node._left is not None:
            raise ValueError('left child of p already exists')
        else:
            node._left = self._Node(e)
            self._size += 1
        return self._make_position(node._left)

    def _add_right(self, p, e):
        node = self._validate(p)
        if node._right is not None:
            raise ValueError('left child of p already exists')
        else:
            node._right = self._Node(e)
            self._size += 1
        return self._make_position(node._right)

    def _replace(self, p, e):
        node = self._validate(p)
        old_element = node._element
        node._element = e
        return old_element

    def _delete(self, p):
        node = self._validate(p)
        if self.num_children(p) == 2:
            raise ValueError('p has 2 children')
        child = node._left if node._left is not None else node._right
        if child is not None:
            child._parent = node._parent
            if node == self._root:
                self._root = child
            else:
                parent = node._parent
                if node is parent._left:
                    parent._left = child
                else:
                    parent._right = child
        self._size -= 1
        return node._element

    def _attach(self, p, t1, t2):
        if not self.is_leaf(p):
            raise ValueError('p must be a leaf')
        if not type(self) is type(t1) is type(t2):
            raise TypeError('t1, t2 and self must be match')
        node = self._validate(p)
        self._size += (len(t1) + len(t2))
        if not t1.is_empty():
            node._left = t1._root
            t1._root._parent = node
            t1._root = None
            t1._size = 0
        if not t2.is_empty():
            node._right = t2._root
            t2._root._parent = node
            t2._root = None
            t2._size = 0


class MutableLinkedBinaryTree(LinkedBinaryTree):
    def add_root(self, e):
        return LinkedBinaryTree._add_root(self, e)

    def add_left(self, p, e):
        return LinkedBinaryTree._add_left(self, p, e)


if __name__ == '__main__':
    print(LinkedBinaryTree.__abstractmethods__)
    T = MutableLinkedBinaryTree()
