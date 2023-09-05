
class AST(object):
    pass

class FUNC(AST):
    def __init__(self, token, name, var_param, body):
        self.token = token
        self.name = name.value
        self.body= body
        self.var_param=var_param
        self.val_param=[]

class Param(AST):
    def __init__(self, token, val_param):
        self.token = token
        self.val_param=val_param

class Call(AST):
    def __init__(self, token, val_param, name):
        self.token = token
        self.name = name.value
        self.val_param = val_param

class Return(AST):
    def __init__(self, token, value):
        self.token = token
        self.value = value

class Compound(AST):
    # Compound AST node represents a compound statement. 
    # It contains a list of statement nodes in its children variable.
    #Represents a 'BEGIN ... END' block
    def __init__(self):
        self.children = []

class BinaryOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = op
        self.op = op
        self.right = right

class UnaryOp(AST):
    def __init__(self, op, expr):
        #expr is an AST node here, remeber it has to be one so that visit_
        self.token = op
        self.op = op
        self.expr = expr

class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class Variable(AST):
    """The Variable node is constructed out of ID token."""
    def __init__(self, token):
        self.token = token
        self.value = token.value


class FOR_Block(AST):
    def __init__(self, token, assign, cond, change, body):
        self.token = token
        self.assign = assign
        self.cond = cond
        self.change = change
        self.body = body

class IF_Block(AST):
    def __init__(self, cond, token, if_body, elseif_nodes, else_body):
        self.cond = cond
        self.token = token
        self.if_body = if_body
        self.elseif_nodes = elseif_nodes
        self.else_body = else_body

class ELSEIF_Block(AST):
    def __init__(self, cond, token, body):
        self.cond = cond
        self.token = token
        self.body = body

class Print(AST):
    def __init__(self, token, content):
        self.token = token
        self.content = content 

class NoOp(AST):
    pass
