from Keywords import *
from AST import *

#LEXER
#Starting part of the Lexer/Tokenizer
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

class Error:
    def __init__(self):
        self.res=[]

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
        self.col=0
        self.err=Error()
        self.current_char=self.text[self.pos]

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
        self.err.res.append('DISSONANCE DETECTED!!' + '\n Wrong note:'+self.current_char+'\n at line: '+ str(self.line))
        self.advance()   

    def get_string(self):
        self.advance()
        value=""
        while self.current_char is not None and self.current_char is not "\"":
            value+=self.current_char
            self.advance()
        self.advance()
        return str(value)


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