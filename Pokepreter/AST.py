class AST(object):
    pass

class BinaryOp(AST):
    def __init__(self,left,op,right):
        self.left=left
        self.token=self.op=op
        self.right=right

class Num(AST):
    def __init__(self,token):
        self.token=token
        self.value=token.value    

class UnaryOp(AST):
    def __init__(self,op,expr):
        self.token=self.op=op
        self.expr=expr  

class Compound(AST):
    def __init__(self):
        self.children=[]


class Assign(AST):
    def __init__(self,left,op,right):
        self.left=left
        self.token=self.op=op
        self.right=right    

class Variable(AST):
    def __init__(self,token):
        self.token=token
        self.value=token.value        

class Return(AST):
    def __init__(self,token,value):
        self.token=token
        self.value=value      

class FUNC(AST):
    def __init__(self,token,name,var_param,body):
        self.token=token
        self.name=name.value
        self.var_param=var_param
        self.body=body   
        self.param_value=[]       

class Call(AST):
    def __init__(self,token,val_param,name):
        self.token=token
        self.name=name
        self.val_param=val_param  

class FOR_Block(AST):
    def __init__(self,token,assign,cond,oper,body):
        self.token=token
        self.assign=assign
        self.cond=cond
        self.oper=oper
        self.body=body

class IF_Block(AST):
    def __init__(self,token,cond,if_body,elseif_body,else_body):
        self.token=token
        self.cond=cond
        self.if_body=if_body
        self.elseif_body=elseif_body
        self.else_body=else_body

class ELSEIF_Block(AST):
    def __init__(self,token,cond,body):
        self.token=token
        self.cond=cond
        self.body=body

class PRINT(AST):
    def __init__(self,token,expressions):
        self.token=token
        self.expressions=expressions

class NoOp(AST):
    pass                         
