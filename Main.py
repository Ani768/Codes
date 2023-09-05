import operator
from Interpreter import *


""" 
Rhythmcoder 
           -Anirudh S
           
  
    GRAMMAR:
        program: 
            Rhythmcoder is a programming language that was build entirely from scratch in a week. There are no additional libraries that need to be installed for the language. 
It includes a lot of functionalities that a normal programming lanuguage has. Some of it's features include:
1) Local Namespace handling
2) Recursive function calls
3) Unique music based syntax
4) Error handling + special music prompts
5) Basic mathematical expressions and operations
6) looping and conditional constructs(For and if-else if -else)
7) Functions with parameters and calls

Below is the grammar of Rhythmcoder.     

        func : 
            CHORUS ID (param) compound_statement

        RESOLVE exp|empty 
        
        param: 
            empty|(expr COMMA)*
        
        compound_statement: 
            BEGIN statement_list END
        
        statement_list : 
            (
                  assignment_statement 
                | empty
                | if_block
                | print 
                | for_block
                |expr
                |return 
            )*
        
        assignment_statement : 
            variable ASSIGN expr 
        
        empty :
        
        print : 
            PLAY( ((str|expr)COMMA)*  )  
        
        call : 
            TRACK ID (param)

        if_block: 
            VERSE expr 
                compound_statement 
            | (INTERLUDE
                compound_statement)*
            | BRIDGE
                compount_statement
            | empty

    
        elseif_block:
            INTERLUDE expr 
                compound_statement 
    
        for_block: 
            LOOP(assignment_statement, cond, change,) 
                compounf_statement


        factor :
             PLUS factor
           | MINUS factor
           | INTEGER
           | LPAREN expr RPAREN
           | variable
           | call

        term: 
            factor (
                        (
                             MUL 
                            | DIV 
                            | LESS_THAN 
                            | LESS_THAN_EQUAL 
                            | EQUALS 
                            | GREATER_THAN 
                            | GREATER_THAN_EQUAL
                        ) 
                    factor  
                    )*

        expr: 
            term (
                    (
                          PLUS 
                        | MINUS
                    ) 
                 term
                 )*

    
        variable: 
            ID 
"""




def main():
        text = """
          CHORUS foo(a,b,c,){
            PLAY("Sum of three numbers is: ",a*b*c,"\n",);
             RESOLVE a*b*c;
           }

          CHORUS SONG(){
            p = TRACK foo(3,4,);
            VERSE(p<50)
            {
           PLAY("Product of three numbers is: ",p,"\n",);
           }
           }
                                     
     """
        lexer = Lexer(text)
        parser = Parser(lexer)
        interpreter = Interpreter(parser)
        interpreter.interpret()


      

if __name__=='__main__':
    main()        
    
