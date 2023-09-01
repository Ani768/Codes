from LEXER import *
from AST import *
from Keywords import *

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
            self.error(token_type,self.current_token.type)
            self.eat(self.current_token.type)
            return

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
        node=self.empty()
        result=[node]
        while self.current_token.type in(ID,IF,PRINT,FOR,CALL,RETURN):
            if self.current_token.type==ID:
                node=self.assignment_statement()
                self.eat(SEMI)
            elif self.current_token.type == IF:
                node= self.if_block()
            elif self.current_token.type== FOR:
                node=self.for_block()
            elif self.current_token.type== PRINT:
                node=self.print()
                self.eat(SEMI)
            elif self.current_token.type == CALL:
                node=self.call()
                self.eat(SEMI)
            elif self.current_token.type ==RETURN:
                node=self.ret()
                self.eat(SEMI)                       
            result.append(node)    

        return result      
     
    def print(self):
        token= self.current_token
        node=self.empty()
        result=[node]
        self.eat(PRINT)
        self.eat(LPAREN)
        while self.current_token.type in (STR,ID,PLUS,MINUS,INT,LPAREN,CALL):
            if self.current_token.type == STR:
                node=self.current_token.value
                self.eat(STR)
            elif self.current_token.type in(ID,PLUS,MINUS,INT,LPAREN,CALL):
                node=self.expr()
            result.append(node)        
            self.eat(COMMA)  

        self.eat(RPAREN)
        return PRINT(token,result)

    def call(self):
        token=self.current_token
        self.eat(CALL)
        name=self.variable()
        self.eat(LPAREN)
        val_param=self.param(2)
        self.eat(RPAREN)

        return Call(token, val_param, name)         

    def if_block(self):
        token=self.current_token
        self.eat(IF)
        cond=self.expr()
        if_body=self.compound_statement()
        else_if=[]
        while(self.current_token.type == ELSEIF):
            else_if.append(self.else_if_block())

        if(self.current_token.type== ELSE):
            self.eat(ELSE)
            else_body=self.compound_statement()
        else:
            else_body=self.empty()

        node= IF_Block(token,cond,if_body,else_if,else_body)
        return node

    def else_if_block(self):
        token=self.current_token
        self.eat(ELSEIF)
        cond=self.expr()
        body=self.compound_statement()
        node=ELSEIF_Block(token,cond,body)
        return node

    def for_block(self):
        token=self.current_token
        self.eat(FOR)
        self.eat(LPAREN)
        val=self.assignment_statement()
        self.eat(SEMI)
        cond=self.expr()
        self.eat(SEMI)
        oper=self.assignment_statement()
        self.eat(SEMI)
        self.eat(RPAREN)
        body=self.compound_statement()
        return FOR_Block(token,val,cond,oper,body)










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
            self.error(EOF,self.current_token.type)
        return node    
