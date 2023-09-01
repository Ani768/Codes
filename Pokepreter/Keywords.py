from LEXER import *
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
ELSE               =   'INTERLUDE'
ELSEIF             =   'BRIDGE'
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