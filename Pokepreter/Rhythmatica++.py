import operator
from LEXER import *
from Keywords import *
from AST import *
from PARSER import *

INT            =   'INT'

# Operators
PLUS               =   'PLUS'
MINUS              =   'MINUS'
MULTIPLY           =  'MULTIPLY'
DIVIDE             = 'DIVIDE'
EOF                =   'EOF'
LPAREN             =   '('
RPAREN             =   ')'
LESS_THAN          =   'LESS_THAN'
LESS_THAN_EQUAL    =   'LESS_THAN_EQUAL'
EQUALS             =   'EQUALS'
GREATER_THAN       =   'GREATER_THAN'
GREATER_THAN_EQUAL =   'GREATER_THAN_EQUAL'

# Key Words
BEGIN              =   '{'
END                =   '}'
IF                 =   'VERSE'
FI                 =   'FI'
ELSE               =   'INTERLUDE'
ELSEIF             =   'BRIDGE'
FOO                =   'FOO'
PRINT              =   'PLAY'
FOR                =   'LOOP'
FUNC               =   'CHORUS'
CALL               =   'CALL'
RETURN             =   'RESOLVE'

#Misc Tokens
SEMI               =   'SEMI'
DOT                =   'DOT'
ASSIGN             =   'ASSIGN'
ID                 =   'ID'
STR                =   'STR'
COMMA              =   ','

class Error:
    def __init__(self):
        self.res=[]

class Token(object):
    def __init__(self, type, value,line=0):
        self.type = type
        self.value = value
        self.line = line

    def __str__(self):
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=str(self.value)
        )
        
RESERVED_KEYWORDS = {
    'BEGIN': Token('BEGIN', '{'),
    'END': Token('END', '}'),
    'IF': Token('IF', 'VERSE'),
    'FI': Token('FI', 'FI'),
    'ELSE': Token('ELSE', 'INTERLUDE'),
    'ELSEIF': Token('ELSEIF', 'BRIDGE'),
    'PRINT': Token('PRINT', 'PLAY'),
    'FOR'  : Token('FOR', 'LOOP'),
    'FUNC'  : Token('FUNC', 'CHORUS'),
    'CALL'  : Token('CALL', 'CALL'),
    'RETURN'  : Token('RETURN', 'RESOLVE'),
}


class Token(object):
    def __init__(self,type,value,line=0):
        self.type=type
        self.value=value
        self.line=line

    def __str__(self):
        return 'Token({type},{value})'.format(
            type=self.type,
            value=repr(self.value)
        )    

    def __repr__(self):
        return self.__str__()

#These are the reserved keywords set for the language
RESERVED_KEYWORDS = {
    'BEGIN': Token('BEGIN','{'),
    'END': Token('END', '}'),
}


#LEXER
#Starting part of the Lexer/Tokenizer
class Lexer(object):

    def peek(self):
        peek_pos=self.pos+1
        if(peek_pos>len(self.text)-1):
            return None
        else:
            return self.text[peek_pos]    

    def __init__(self,text):
        self.text=text
        self.pos=0
        self.line=0
        self.err=[]
        self.current_char=text[self.pos]

    def advance(self):
        self.pos+=1
        if self.pos>len(self.text)-1 :
            self.current_char=None
        else:
            self.current_char=self.text[self.pos]    

    def skip_space(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()


    def integer(self):
        value=''
        while self.current_char is not None and self.current_char.isdigit():
            value+=self.current_char
            self.advance()
        return int(value)


    def _id(self):
          result = ''
          while self.current_char is not None and self.current_char.isalnum():
           result += self.current_char
           self.advance()

          token = RESERVED_KEYWORDS.get(result, Token(ID, result))
          token.line=self.line
          return token
           

    def error(self):
        self.err.append('DISSONANCE DETECTED!!' + '\n Wrong note:'+self.current_char+'\n at line: '+ str(self.line))
        self.advance()   

    def get_string(self):
        self.advance()
        value=""
        while self.current_char is not None and self.current_char is not "\"":
            value+=self.current_char
            self.advance()
        self.advance()
        return value


    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_space()
                continue

            if self.current_char.isdigit():
                return Token(INT, self.integer())

            if self.current_char.isalpha():
                return self._id()  


            if self.current_char == '=':
                self.advance()
                return Token(ASSIGN,'=',self.line)     

            if self.current_char == '+':
                self.advance()
                return Token(PLUS,'+',self.line)

            if self.current_char == ';':
                self.advance()
                return Token(SEMI,';',self.line)   

            if self.current_char == '-':
                self.advance()
                return Token(MINUS,'-',self.line)

            if self.current_char == '*':
                self.advance()
                return Token(MULTIPLY,'*',self.line)

            if self.current_char =='{':
                self.advance()
                return Token(BEGIN,'{',self.line)

            if self.current_char == '}':
                self.advance()
                return Token(END,'}',self.line)        

            if self.current_char == '/':
                self.advance()
                return Token(DIVIDE,'/',self.line) 

        
            if self.current_char == '=' and self.peek() == '=':
                self.advance()
                return Token(EQUALS, '==',self.line)

            if self.current_char == '=':
                self.advance()
                return Token(ASSIGN, '=',self.line)

            if self.current_char == '<':
                self.advance()
                return Token(LESS_THAN,'<',self.line)    

            if self.current_char == '>':
                self.advance()
                return Token(LESS_THAN,'>',self.line)       

            if self.current_char == '<' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(LESS_THAN_EQUAL, '<=',self.line)

            if self.current_char == '>' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(GREATER_THAN_EQUAL, '>=',self.line)    

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN,'(',self.line)

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN,')',self.line)        

            self.error()

        return Token(EOF, None)


#Ending of the lexer part of the code



#Beginning of the definition of AST's
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

class NoOp(AST):
    pass                         

#End of the classes for Abstract Syntax Trees(AST's)


#Starting of the PARSER class
class Parser(object):
    
    def __init__(self,lexer):
        self.lexer=lexer
        self.current_token=self.lexer.get_next_token() 

    def error(self,expected,detected):
        self.lexer.err.res.append('DISSONANCE DETECED!! This is not jazz, do not change the scale mate!'
                        +'\n      Expected: '+expected
                        +'\n      Recieved: '+ detected
                        +'\n      At line:  '+str(self.lexer.line)
                        )

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token=self.lexer.get_next_token()
        else:
            self.error()

    def program(self):
        node = self.compound_statement()
        self.eat(EOF)
        return node     


    def func(self):
        token=self.current_token
        self.eat(FUNC)
        name=self.variable()
        self.eat(LPAREN)
        var_param=self.param(1)
        self.eat(RPAREN)
        body =self.compound_statement()       
        return FUNC(token,name,var_param,body)

    def param(self, expected):
        token=self.current_token
        flag1=0
        flag2=0
        value=[]
        token=self.current_token
        while token.type in (ID,PLUS,MINUS,LPAREN,INT,CALL):
            if expected==1:
                flag1=1
                node=self.variable()
            elif expected==2:
                flag2=1
                node=self.expr()
            value.append(node)
            self.eat(COMMA)
            if(flag1+flag2==2):
                raise Exception("HARMONY MISMATCH DETECTED!!Two different melodies at the same time? Please keep away from polyrhythms!")
            elif(flag1+flag2==0):
                return value
            else:
                if(expected==1 and flag1==1):
                    return value
                elif (expected==2 and flag2==1):
                    return value
                else:
                    raise Exception('It seems you have not matched the melody which the song expects. Please match the melody and try again!')     

    def ret(self):
        token=self.current_token
        self.eat(RETURN)
        while token.type in (PLUS,MINUS,ID,LPAREN,INT,CALL):
            node=self.expr()
            return Return(token,node)
        node=NoOp()
        return Return(token,node)    



    def compound_statement(self):
        self.eat(BEGIN)
        nodes= self.statement_list()    
        self.eat(END) 

        root =Compound()
        for node in nodes:
            root.children.append(node)
        return root    

    def statement_list(self):
        node=self.statement()

        result=[node]

        while self.current_token.type ==  SEMI:
            self.eat(SEMI)
            result.append(self.statement())    

        if self.current_token.type== ID:
            self.error()

        return result      

    def statement(self):
        if self.current_token.type== BEGIN:
            node=self.compound_statement()
        elif self.current_token.type==ID:
            node=self.assignment_statement()
        else:
            node=self.empty()
        return node      

    def assignment_statement(self):
        left=self.variable()
        token=self.current_token
        self.eat(ASSIGN)
        right=self.expr()
        node=Assign(left,token,right)
        return node

    def variable(self):
        node=Variable(self.current_token)
        self.eat(ID)
        return node

    def empty(self):
        return NoOp()    

    def term(self):
        token=self.current_token
        if token.type==PLUS:        
            self.eat(PLUS)
            node=UnaryOp(token, self.term())
            return node
        elif token.type==MINUS:
            self.eat(MINUS)
            node=UnaryOp(token, self.term())
            return node
        elif token.type==INT:
            self.eat(INT)
            node=Num(token) 
            return node
        elif token.type==LPAREN:       
            self.eat(LPAREN)
            result=self.expr()
            self.eat(RPAREN)
            return result
        else:
            node=self.variable()
            return node


    def priority_1(self):
        node=self.term()
        while self.current_token.type in (MULTIPLY,DIVIDE,LESS_THAN, LESS_THAN_EQUAL, EQUALS, GREATER_THAN, GREATER_THAN_EQUAL):
            token=self.current_token
            if token.type == MULTIPLY:
                self.eat(MULTIPLY)
            elif token.type == DIVIDE:
                self.eat(DIVIDE)
            elif token.type == LESS_THAN:
                self.eat(LESS_THAN)
            elif token.type == GREATER_THAN:
                self.eat(GREATER_THAN)
            elif token.type == EQUALS:
                self.eat(EQUALS)
            elif token.type == GREATER_THAN_EQUAL:
                self.eat(GREATER_THAN_EQUAL)
            elif token.type ==  LESS_THAN_EQUAL:
                self.eat(LESS_THAN_EQUAL)                    
            node = BinaryOp(left=node, op=token, right=self.term() )
        return node
    

    def expr(self):
        node=self.priority_1()
        while self.current_token.type in (PLUS,MINUS):
            token=self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type== MINUS:
                self.eat(MINUS)

            node =  BinaryOp(left=node, op=token, right=self.priority_1()) 
        return node          

    def Parse(self):
        node=self.program()
        if self.current_token.type != EOF:
            self.error()
        return node    

#Ending of PARSER


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

                     
#Main which implements the REPL loop(Read-Evaluate-Print_Loop)
def main():
    while True:
        try:
            text=input('calc>')
        except EOFError:
            break
        if not text:
            continue
        lexer=Lexer(text)
        parser=Parser(lexer)
        interpretr=Interpreter(parser)
        interpretr.interpret_and_print_variables()

if __name__=='__main__':
    main()        
    


