import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 #at initializtion, op is not affected by Value object
        self._backward = lambda: None # None for leaf nodes in the graph
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        """
        ^ to work with non-value objects by wrapping them around Value and thereby providing them a data attribute
        eg: to perform a + 1 (where 1 is not a Value object)
        """
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            """
            using += instead of + to accumulate gradients instead of overwriting (in the case of adding a node to itself)
            """
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
 
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), '**{other}')
        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other
  
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        x = self.data
        out = Value(x if x > 0 else 0, (self, ), 'relu')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        out = Value(1/(1 + math.exp(-x)), (self, ), 'sigmoid')
        
        def _backward():
            self.grad += (1 - out.data) * out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        print(topo)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
