from PARSE import *
from AST import *
from Keywords import *
#The declaration of the Nodevisitor method to interpret the AST's based on the type 
#Starting of the NodeVisitor Method
class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))
#End of the NodeVisitor method


#Starting of the INTERPRETER  that is used to interpret the parsed statements
class Interpreter(NodeVisitor):
    def __init__(self, parser):
        self.GLOBAL_SCOPE = {}
        self.parser = parser

    def visit_BinaryOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MULTIPLY:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIVIDE:
            return self.visit(node.left) / self.visit(node.right)
        elif node.op.type == LESS_THAN:
            return self.visit(node.left) < self.visit(node.right)    
        elif node.op.type == GREATER_THAN:
            return self.visit(node.left) > self.visit(node.right)  
        elif node.op.type == EQUALS:
            return self.visit(node.left) == self.visit(node.right)      
        elif node.op.type == LESS_THAN_EQUAL:
            return self.visit(node.left) <= self.visit(node.right)      
        elif node.op.type == GREATER_THAN_EQUAL:
            return self.visit(node.left) >= self.visit(node.right)                 

    def visit_UnaryOp(self, node):
        if node.op.type == PLUS:
            return +self.visit(node.expr)
        elif node.op.type ==  MINUS:
            return -self.visit(node.expr)    

    def visit_NoOp(self,node):
        pass

    def visit_Compound(self,node):
        for child in node.children:
            self.visit(child)

    def visit_Assign(self,node):
        var_name=node.left.value
        self.GLOBAL_SCOPE[var_name] = self.visit(node.right)        

    def visit_Num(self, node):
        return node.value

    def visit_Variable(self,node):
        var_name=node.value
        val=self.GLOBAL_SCOPE.get(var_name)
        if val is None:
            raise NameError(repr(var_name))
        else:
            return val    

    def interpret(self):
        tree = self.parser.Parse()
        return self.visit(tree)    

    def interpret_and_print_variables(self):
        tree = self.parser.Parse()
        self.visit(tree)
        for var_name, val in self.GLOBAL_SCOPE.items():
            print(f"{var_name}: {val}")             
#Ending of the INTERPRETER     
