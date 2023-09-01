import operator
from LEXER import *
from Keywords import *
from AST import *
from PARSE import *
from interpret import *


                     
#Main which implements the REPL loop(Read-Evaluate-Print_Loop)
def main():
     while True:
        text = input('YourPromptHere> ')
        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        interpreter.interpret()


      

if __name__=='__main__':
    main()        
    


