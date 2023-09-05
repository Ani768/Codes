import io
import sys
from io import StringIO
from Parse import *

def get_musical_error_message(num_errors):
    musical_prompts = [
        "One off Note Detected 🎵",
        "Prepare for trouble and make it double!! 🎵🎵",
        "Oops! A Minor Chord 🎶",
        "A Single Rest in the Melody 🎼",
        "A pentatonic scale of mistakes 🎶🎶",
        "Six Strikes, You're Almost Completely Out of Tune! 🎶🎶🎶",
        "A Symphony of Mistakes! 🎶🎶🎶🎶🎶",
        "An Orchestra of Errors 🎻🎹🎷🥁🎤",
        "Time for a Musical Tune-Up! 🎵🎼🎶🎺🪕",
    ]
    if num_errors < 1:
        return "No Errors Detected, Harmony Prevails! 🎵🌟"
    elif num_errors >= len(musical_prompts):
        return f"An Epic Symphony of {num_errors} Errors! 🎵🎼🎶🎺🪕"
    else:
        return musical_prompts[num_errors - 1]
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
    GLOBAL_CHORUS_NAMES = {}
    CUR_CHORUS = ""
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
        elif node.op.type == LESS_THAN_EQUAL:
            return self.visit(node.left) <= self.visit(node.right)
        elif node.op.type == EQUALS:
            return self.visit(node.left) == self.visit(node.right)
        elif node.op.type == GREATER_THAN:
            return self.visit(node.left) > self.visit(node.right)
        elif node.op.type == GREATER_THAN_EQUAL:
            return self.visit(node.left) >= self.visit(node.right)
        else:
            raise Exception('Invalid operator')

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
        if (var_name in self.GLOBAL_CHORUS_NAMES):
            self.parser.lexer.err.res.append(
                ' Melodies off on the CHORUS, MELODY ERROR!!' +
                '\n      COPYRIGHTS CLAIMED FOR THE SONG!!' +
                '\n      At line:    ' + str(node.left.token.line) +
                '\n      Wrong char: ' + node.left.token.value)
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

    def visit_ELSEIF_Block(self, node):
        if self.visit(node.cond):
            self.visit(node.body)
            return 1

    def visit_FOR_Block(self, node):
        self.visit(node.assign)
        while self.visit(node.cond):
            self.visit(node.body)
            self.visit(node.change)

    def visit_Print(self, node):
        for n in node.content:
            if (type(n) == str):
                print(n, end="")
            elif (type(n) == NoOp):
                pass
            else:
                print(self.visit(n), end="")

    def visit_Call(self, node):
        self.GLOBAL_CHORUS_NAMES[node.name].val_param = node.val_param
        self.gf = 0
        self.visit(self.GLOBAL_CHORUS_NAMES[node.name])
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
        self.CUR_CHORUS = node.name    

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
        tree = self.parser.parse()
        for n in tree:
            self.GLOBAL_CHORUS_NAMES[n.name] = n

        if ("SONG" not in self.GLOBAL_CHORUS_NAMES):
            self.parser.lexer.err.res.append(
                ' Invalid Song Name – COMPOSITION ERROR\n' +
                ' Song does not exist')
        else:
            self.GLOBAL_CHORUS_NAMES["SONG"].val_param.append(NoOp())
            self.visit(self.GLOBAL_CHORUS_NAMES["SONG"])
            self.res = new_stdout.getvalue()

        # we need to restore std_out at this point
        sys.stdout = old_stdout
        errObj = self.parser.lexer.err

        # Self Explanatory, should I print res, or display error
        if (len(errObj.res) == 0):
            print(self.res)
            print('--SONG PLAYED IN PERFECT HARMONY!!!')
        else:
            print('-The SONG has slight dissonances , about ' + str(len(errObj.res)) +
                  ' in total!!!!')
            error_message = get_musical_error_message(len(errObj.res))
            print(error_message)
            for i in range(len(errObj.res)):
                print(i + 1, ": ")
                print(errObj.res[i])
            