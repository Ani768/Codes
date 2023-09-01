###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################
from PARSE import *
from Keywords import *
from LEXER import *
from AST import *
import io
import sys
from io import StringIO

# captures and changes std_out
# useful to not display ans, when compilation has errors
old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout


class NodeVisitor(object):
    # tells if im breaking cos of return statement, reset before every function_call
    gf = 0
    # stores the return value, 0 is the standard return value
    ans = 0

    def visit(self, node, ret=[]):
        if (self.gf == 1 and type(node).__name__ != "Compound"):
            # If the type was Compound, we want to enter it, so that we can erase the local variables
            return self.ans

        method_name = 'visit_' + type(node).__name__
        # if you forgot what this does, google getattr method of python3
        visitor = getattr(self, method_name, self.generic_visit)

        if (type(node).__name__ == "Compound"):
            # we want to make the compound statement declare the variables
            # after its '{' brackets is called, to make the paramteres local to the function
            return visitor(node, ret)
        else:
            return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor):
    GLOBAL_SCOPE = {}
    CUR_SCOPE = []
    GLOBAL_FUNC_NAMES = {}
    CUR_FUNC = ""
    cnt = -1

    def __init__(self, parser):
        self.parser = parser
        res = ""

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

    def visit_Num(self, node):
        return node.value

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == PLUS:
            return +self.visit(node.expr)
        elif op == MINUS:
            return -self.visit(node.expr)

    def visit_Compound(self, node, ret=[]):
        self.CUR_SCOPE.append(set())
        self.cnt += 1

        for i in ret:
            self.visit(i)
        for child in node.children:
            self.visit(child)
            # stop visiting children, if i have encountered a return statement in function
            if self.gf == 1: break

        for v in self.CUR_SCOPE[self.cnt]:
            self.GLOBAL_SCOPE[v].pop()
            if len(self.GLOBAL_SCOPE[v]) == 0:
                del self.GLOBAL_SCOPE[v]

        self.CUR_SCOPE.pop()
        self.cnt -= 1

    def visit_NoOp(self, node):
        pass

    def visit_Assign(self, node):
        var_name = node.left.value
        val = self.visit(node.right)
        if (var_name in self.GLOBAL_FUNC_NAMES):
            self.parser.lexer.err.res.append(
                ' Invalid Function Naming -- COMPILE TIME ERROR' +
                '\n      There is a function with the same name' +
                '\n      At line:    ' + str(node.left.token.line))
        else:
            if var_name in self.CUR_SCOPE[self.cnt]:
                self.GLOBAL_SCOPE[var_name].pop()
                self.GLOBAL_SCOPE[var_name].append(val)
            else:
                #print(1)
                if var_name not in self.GLOBAL_SCOPE:
                    self.GLOBAL_SCOPE[var_name] = list()
                self.GLOBAL_SCOPE[var_name].append(val)
                #print(2,var_name, self.GLOBAL_SCOPE[var_name])
            #print("before: ", self.CUR_SCOPE)
            self.CUR_SCOPE[self.cnt].add(var_name)
            #print("after: ", self.CUR_SCOPE)

    def visit_Variable(self, node):
        var_name = node.value
        val = self.GLOBAL_SCOPE.get(var_name)[-1]
        if val is None:
            raise NameError(repr(var_name))
        else:
            return val

    def visit_IF_Block(self, node):
        if self.visit(node.cond):
            self.visit(node.if_body)
            return

        for n in node.elseif_nodes:
            ok = self.visit(n)
            if ok: return

        self.visit(node.else_body)

    def visit_ElSEIF_Block(self, node):
        if self.visit(node.cond):
            self.visit(node.body)
            return 1

    def visit_FOR_Block(self, node):
        self.visit(node.assign)
        while self.visit(node.cond):
            self.visit(node.body)
            self.visit(node.change)

    def visit_PRINT(self, node):
        for n in node.expressions:
            if (type(n) == str):
                print(n, end="")
            elif (type(n) == NoOp):
                pass
            else:
                print(self.visit(n), end="")

    def visit_Call(self, node):
        self.GLOBAL_FUNC_NAMES[node.name].val_param = node.val_param
        self.gf = 0
        self.visit(self.GLOBAL_FUNC_NAMES[node.name])
        self.gf = 0
        return self.ans

    def visit_Return(self, node):
        # update teh global return value
        self.ans = self.visit(node.value)
        # set return flag to 1
        self.gf = 1

        if (self.ans >= 0): return
        if node.value in self.GLOBAL_SCOPE:
            self.ans = self.GLOBAL_SCOPE[node.value][-1]
        else:
            self.ans = 0
        return

    def visit_FUNC(self, node):
        # update cur_func name for debugging
        self.CUR_FUNC = node.name

        # we have to declare the parameters of a function, after the compound statement of the
        # function is called. So we prepare the list ret and pass it as param to the visit_Compound
        # which will take care from there on
        ret = []
        for i in range(len(node.var_param)):
            ret.append(
                Assign(node.var_param[i], Token('ASSIGN', 'ASSIGN'),
                       node.val_param[i]))

        val = self.visit(node.body, ret)
        return val

    def interpret(self):

        tree = self.parser.Parse()
        # Redirect stdout to capture any output
        old_stdout = sys.stdout
        new_stdout = StringIO()
        sys.stdout = new_stdout

        try:
            for n in tree:
                self.GLOBAL_FUNC_NAMES[n.name] = n

            if "MAIN" not in self.GLOBAL_FUNC_NAMES:
                raise Exception("Main Function does not exist")
            else:
                self.GLOBAL_FUNC_NAMES["MAIN"].val_param.append(NoOp())
                self.visit(self.GLOBAL_FUNC_NAMES["MAIN"])
                self.res = new_stdout.getvalue()

        except Exception as e:
            # Handle exceptions here, you can suppress them if needed
            pass

        finally:
            # Restore sys.stdout and close the StringIO
            sys.stdout = old_stdout
            new_stdout.close()

    