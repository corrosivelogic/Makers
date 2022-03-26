import argparse
import sys
from enum import Enum

_SHOULD_LOG_SCOPE = False  # see '--scope' command line option
_SHOULD_LOG_STACK = False  # see '--stack' command line option

#Exception Classes
class ErrorCode(Enum):
    UNEXPECTED_TOKEN = 'Unexpected token'
    ID_NOT_FOUND     = 'Identifier not found'
    DUPLICATE_ID     = 'Duplicate id found'


class Error(Exception):
    def __init__(self, error_code=None, token=None, message=None):
        self.error_code = error_code
        self.token = token
        self.message = f'{self.__class__.__name__}: {message}'


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class SemanticError(Error):
    pass

#  LEXER                                                                      
class TokenType(Enum):
    # single-character tokens
    PLUS          = '+'
    MINUS         = '-'
    MUL           = '*'
    FLOAT_DIV     = '/'
    LPAREN        = '('
    RPAREN        = ')'
    SEMI          = ';'
    DOT           = '.'
    COLON         = ':'
    COMMA         = ','
    LESS_THAN     = '<'
    GREATER_THAN  = '>'
    EQUAL         = '='    
    #reserved words
    PROGRAM       = 'ABJOHYAHIH'  
    INTEGER       = 'INTHYE'
    REAL          = 'REALHYE'
    INTEGER_DIV   = 'DIVCROW'
    VAR           = 'VAR'
    PROCEDURE     = 'yealagblockh'
    BEGIN         = 'SHURUYAHASE'
    WRITE         = 'LIKHO'
    WRITELN       = 'LIKHLINE'
    TRUE          = 'SAHIH'
    FALSE         = 'GALTBAAT'
    BOOLEAN       = 'BOOLEAN'
    IF            = 'AGARHOMIE'
    THEN          = 'FIR'
    ELSE          = 'VARNA'
    END           = 'KHATAMHOGYATATABYEBYE'     
    # misc
    ID            = 'ID'
    INTEGER_CONST = 'INTEGER_CONST'
    REAL_CONST    = 'REAL_CONST'
    ASSIGN        = ':='
    STRING        = 'STRING'
    #composite_character tokens
    LESS_EQUAL    = '<='
    GREATER_EQUAL = '>='
    NOT_EQUAL     = '<>'
    EOF           = 'EOF'    


class Token:
    def __init__(self, type, value, lineno=None, column=None):
        self.type = type
        self.value = value
        self.lineno = lineno
        self.column = column

    def __str__(self):
        return 'Token({type}, {value}, position={lineno}:{column})'.format(
            type=self.type,
            value=repr(self.value),
            lineno=self.lineno,
            column=self.column,
        )

    def __repr__(self):
        return self.__str__()


def _build_reserved_keywords():
    tt_list = list(TokenType)
    start_index = tt_list.index(TokenType.PROGRAM)
    end_index = tt_list.index(TokenType.END)
    reserved_keywords = {
        token_type.value: token_type
        for token_type in tt_list[start_index:end_index + 1]
    }
    return reserved_keywords


RESERVED_KEYWORDS = _build_reserved_keywords()


class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]
        self.lineno = 1
        self.column = 1

    def error(self):
        s = "Lexer error on '{lexeme}' line: {lineno} column: {column}".format(
            lexeme=self.current_char,
            lineno=self.lineno,
            column=self.column,
        )
        raise LexerError(message=s)

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        if self.current_char == '\n':
            self.lineno += 1
            self.column = 0

        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  #end of input
        else:
            self.current_char = self.text[self.pos]
            self.column += 1

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def skip_comment(self):
        while self.current_char != '}':
            self.advance()
        self.advance()  #closing curly brace

    def number(self):
        token = Token(type=None, value=None, lineno=self.lineno, column=self.column)

        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char == '.':
            result += self.current_char
            self.advance()

            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()

            token.type = TokenType.REAL_CONST
            token.value = float(result)
        else:
            token.type = TokenType.INTEGER_CONST
            token.value = int(result)

        return token

    def _id(self):
        """Handle identifiers and reserved keywords"""

        # Create a new token with current line and column number
        token = Token(type=None, value=None, lineno=self.lineno, column=self.column)

        value = ''
        while self.current_char is not None and self.current_char.isalnum():
            value += self.current_char
            self.advance()

        token_type = RESERVED_KEYWORDS.get(value.upper())
        if token_type is None:
            token.type = TokenType.ID
            token.value = value
        else:
            # reserved keyword
            token.type = token_type
            token.value = value.upper()

        return token
    
    def string(self):
        result = ''
        # Create a new token with current line and column number
        token = Token(type=TokenType.STRING, value=None, lineno=self.lineno, column=self.column)
        while self.current_char is not None and self.current_char != "'":
            result += self.current_char
            self.advance()
        
        self.advance() # skip the closing "'"
        
        token.value = result

        return token

    def get_next_token(self):
        #This method is responsible for breaking a sentence apart into tokens. One token at a time.
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '{':
                self.advance()
                self.skip_comment()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return self.number()
            
            if self.current_char == "'":
                self.advance() 
                return self.string()

            if self.current_char == ':' and self.peek() == '=':
                token = Token(
                    type=TokenType.ASSIGN,
                    value=TokenType.ASSIGN.value,
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                self.advance()
                return token
            # composed-character token
            if self.current_char == '<' and self.peek() == '=':
                token = Token(
                    type=TokenType.LESS_EQUAL,
                    value=TokenType.LESS_EQUAL.value, 
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance() 
                self.advance() 
                return token

            if self.current_char == '<' and self.peek() == '>':
                token = Token(
                    type=TokenType.NOT_EQUAL,
                    value=TokenType.NOT_EQUAL.value,
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance() 
                self.advance()
                return token
            
            if self.current_char == '>' and self.peek() == '=':
                token = Token(
                    type=TokenType.GREATER_EQUAL,
                    value=TokenType.GREATER_EQUAL.value, 
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance() 
                self.advance() 
                return token
            # single-character token
            try:
                token_type = TokenType(self.current_char)
            except ValueError:
                self.error()
            else:
                token = Token(
                    type=token_type,
                    value=token_type.value,  
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

        return Token(type=TokenType.EOF, value=None)


#region AST Nodes Declaration
class AST:
    pass

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr

class Compound(AST):
    """Represents a 'BEGIN ... END' block"""
    def __init__(self):
        self.children = []

class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Var(AST):
    """The Var node is constructed out of ID token."""
    def __init__(self, token):
        self.token = token
        self.value = token.value

class NoOp(AST):
    pass

class String(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class Program(AST):
    def __init__(self, name, block):
        self.name = name
        self.block = block

class Block(AST):
    def __init__(self, declarations, compound_statement):
        self.declarations = declarations
        self.compound_statement = compound_statement

class VarDecl(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node

class Type(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class Param(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node

class ProcedureDecl(AST):
    def __init__(self, proc_name, formal_params, block_node):
        self.proc_name = proc_name
        self.formal_params = formal_params  # a list of Param nodes
        self.block_node = block_node

class ProcedureCall(AST):
    def __init__(self, proc_name, actual_params, token):
        self.proc_name = proc_name
        self.actual_params = actual_params  # a list of AST nodes
        self.token = token
        self.proc_symbol = None

class WriteStmt(AST):
    def __init__(self):
        self.new_line = False
        self.expressions = []

class Boolean(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class IfStmt(AST):
    def __init__(self, condition):
        self.condition = condition
        self.consequences = [] # list of statements
        self.alternatives = [] 



#  PARSER                                                                     
class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
        self.next_token = self.lexer.get_next_token()

    def get_next_token(self):
        current_token = self.next_token
        self.next_token = self.lexer.get_next_token()
        return current_token

    def error(self, error_code, token):
        raise ParserError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}',
        )

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token,
            )

    def program(self):
        self.eat(TokenType.PROGRAM)
        var_node = self.variable()
        prog_name = var_node.value
        self.eat(TokenType.SEMI)
        block_node = self.block()
        program_node = Program(prog_name, block_node)
        self.eat(TokenType.DOT)
        return program_node

    def block(self):
        declaration_nodes = self.declarations()
        compound_statement_node = self.compound_statement()
        node = Block(declaration_nodes, compound_statement_node)
        return node

    def declarations(self):
        declarations = []

        while self.current_token.type == TokenType.VAR:
            self.eat(TokenType.VAR)
            while self.current_token.type == TokenType.ID:
                var_decl = self.variable_declaration()
                declarations.extend(var_decl)
                self.eat(TokenType.SEMI)

        while self.current_token.type == TokenType.PROCEDURE:
            proc_decl = self.procedure_declaration()
            declarations.append(proc_decl)

        return declarations

    def formal_parameters(self):
        param_nodes = []

        param_tokens = [self.current_token]
        self.eat(TokenType.ID)
        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            param_tokens.append(self.current_token)
            self.eat(TokenType.ID)

        self.eat(TokenType.COLON)
        type_node = self.type_spec()

        for param_token in param_tokens:
            param_node = Param(Var(param_token), type_node)
            param_nodes.append(param_node)

        return param_nodes

    def formal_parameter_list(self):
        if not self.current_token.type == TokenType.ID:
            return []

        param_nodes = self.formal_parameters()

        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            param_nodes.extend(self.formal_parameters())

        return param_nodes

    def variable_declaration(self):
        var_nodes = [Var(self.current_token)]
        self.eat(TokenType.ID)

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            var_nodes.append(Var(self.current_token))
            self.eat(TokenType.ID)

        self.eat(TokenType.COLON)

        type_node = self.type_spec()
        var_declarations = [
            VarDecl(var_node, type_node)
            for var_node in var_nodes
        ]
        return var_declarations

    def procedure_declaration(self):
        self.eat(TokenType.PROCEDURE)
        proc_name = self.current_token.value
        self.eat(TokenType.ID)
        formal_params = []

        if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            formal_params = self.formal_parameter_list()
            self.eat(TokenType.RPAREN)

        self.eat(TokenType.SEMI)
        block_node = self.block()
        proc_decl = ProcedureDecl(proc_name, formal_params, block_node)
        self.eat(TokenType.SEMI)
        return proc_decl

    def type_spec(self):
        token = self.current_token
        if self.current_token.type == TokenType.INTEGER:
            self.eat(TokenType.INTEGER)
        elif self.current_token.type == TokenType.REAL:
            self.eat(TokenType.REAL)
        elif self.current_token.type == TokenType.BOOLEAN:
            self.eat(TokenType.BOOLEAN)

        node = Type(token)
        return node

    def compound_statement(self): 
        self.eat(TokenType.BEGIN)
        nodes = self.statement_list()
        self.eat(TokenType.END)

        root = Compound()
        for node in nodes:
            root.children.append(node)

        return root

    def statement_list(self):
        node = self.statement()

        results = [node]

        while self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
            results.append(self.statement())

        return results

    def statement(self):
        if self.current_token.type == TokenType.BEGIN:
            node = self.compound_statement()
        elif self.current_token.type in (TokenType.WRITE, TokenType.WRITELN):
            node = self.write_statement()
        elif self.current_token.type == TokenType.ID and self.next_token.type == TokenType.LPAREN:
            node = self.proccall_statement()
        elif self.current_token.type == TokenType.ID and self.next_token.type == TokenType.ASSIGN:
            node = self.assignment_statement()
        elif self.current_token.type == TokenType.ID:
            node = self.variable()        
        elif self.current_token.type == TokenType.IF:
            node = self.if_statement()        
        else:
            node = self.empty()
        return node

    def proccall_statement(self):
        token = self.current_token

        proc_name = self.current_token.value
        self.eat(TokenType.ID)
        self.eat(TokenType.LPAREN)
        actual_params = []
        if self.current_token.type != TokenType.RPAREN:
            node = self.expr()
            actual_params.append(node)

        while self.current_token.type == TokenType.COMMA:
            self.eat(TokenType.COMMA)
            node = self.expr()
            actual_params.append(node)

        self.eat(TokenType.RPAREN)

        node = ProcedureCall(
            proc_name=proc_name,
            actual_params=actual_params,
            token=token,
        )
        return node

    def assignment_statement(self):
        left = self.variable()
        token = self.current_token
        self.eat(TokenType.ASSIGN)
        right = self.expr()
        node = Assign(left, token, right)
        return node

    def variable(self):
        node = Var(self.current_token)
        self.eat(TokenType.ID)
        return node
    
    def write_statement(self):    
        node = WriteStmt()
        if self.current_token.type == TokenType.WRITE:
            self.eat(TokenType.WRITE)            
        elif self.current_token.type == TokenType.WRITELN:
            self.eat(TokenType.WRITELN)
            node.new_line = True
        expressions = []
        if self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)           
            while self.current_token.type != TokenType.RPAREN:
                expressions.append(self.expr())
                while self.current_token.type == TokenType.COMMA:
                    self.eat(TokenType.COMMA)
                    expressions.append(self.expr())
            self.eat(TokenType.RPAREN)
            
        for expression in expressions:
            node.expressions.append(expression)

        return node

    def if_statement(self):
        self.eat(TokenType.IF)
        condition = self.expr()
        self.eat(TokenType.THEN)
        
        consequences = self.statement_list()

        alternatives = []
        
        if self.current_token.type == TokenType.ELSE and self.next_token.type == TokenType.IF:
            alternatives.append(self.elsif_statement())
        elif self.current_token.type == TokenType.ELSE:
            alternatives.extend(self.else_statement())

        node = IfStmt(condition=condition)
        
        for consequence in consequences:
            node.consequences.append(consequence)

        for alternative in alternatives:
            node.alternatives.append(alternative)
        
        return node

    def else_statement(self):
        self.eat(TokenType.ELSE)
        nodes = self.statement_list()
        statements = [
            node for node in nodes
        ]
        return statements
    
    def elsif_statement(self):  
        self.eat(TokenType.ELSE)
        return self.if_statement()

    def empty(self):
        return NoOp()

    def expr(self):
        return self.relation()

    def relation(self):   
        node = self.arithmetic_expr()
        if self.current_token.type in (
            TokenType.LESS_THAN,
            TokenType.GREATER_THAN,
            TokenType.LESS_EQUAL,
            TokenType.GREATER_EQUAL,
            TokenType.EQUAL,
            TokenType.NOT_EQUAL,
        ):
            token = self.current_token
            if self.current_token.type == TokenType.LESS_THAN:
                self.eat(TokenType.LESS_THAN)            
            elif self.current_token.type == TokenType.GREATER_THAN:
                self.eat(TokenType.GREATER_THAN)            
            elif self.current_token.type == TokenType.EQUAL:
                self.eat(TokenType.EQUAL)            
            elif self.current_token.type == TokenType.LESS_EQUAL:
                self.eat(TokenType.LESS_EQUAL)            
            elif self.current_token.type == TokenType.GREATER_EQUAL:
                self.eat(TokenType.GREATER_EQUAL)            
            elif self.current_token.type == TokenType.NOT_EQUAL:
                self.eat(TokenType.NOT_EQUAL)
            node = BinOp(left=node, op=token, right=self.arithmetic_expr())
        return node

    def arithmetic_expr(self):
        node = self.term()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            if token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
            elif token.type == TokenType.MINUS:
                self.eat(TokenType.MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def term(self):
        node = self.factor()

        while self.current_token.type in (
                TokenType.MUL,
                TokenType.INTEGER_DIV,
                TokenType.FLOAT_DIV,
        ):
            token = self.current_token
            if token.type == TokenType.MUL:
                self.eat(TokenType.MUL)
            elif token.type == TokenType.INTEGER_DIV:
                self.eat(TokenType.INTEGER_DIV)
            elif token.type == TokenType.FLOAT_DIV:
                self.eat(TokenType.FLOAT_DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def factor(self):
        token = self.current_token
        if token.type == TokenType.PLUS:
            self.eat(TokenType.PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == TokenType.MINUS:
            self.eat(TokenType.MINUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == TokenType.INTEGER_CONST:
            self.eat(TokenType.INTEGER_CONST)
            return Num(token)
        elif token.type == TokenType.REAL_CONST:
            self.eat(TokenType.REAL_CONST)
            return Num(token)
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        elif token.type == TokenType.STRING:
            self.eat(TokenType.STRING)
            return String(token)
        elif token.type in (TokenType.TRUE, TokenType.FALSE):
            if token.type == TokenType.TRUE:
                self.eat(TokenType.TRUE)
            elif token.type == TokenType.FALSE:
                self.eat(TokenType.FALSE)
            return Boolean(token)
        else:
            node = self.variable()
            return node

    def parse(self):
        node = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token,
            )

        return node


#  AST visitors (walkers)                                                     
class NodeVisitor:
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))
#  SYMBOLS, TABLES, SEMANTIC ANALYSIS                                         
class Symbol:
    def __init__(self, name, type=None):
        self.name = name
        self.type = type
        self.scope_level = 0

class VarSymbol(Symbol):
    def __init__(self, name, type):
        super().__init__(name, type)

    def __str__(self):
        return "<{class_name}(name='{name}', type='{type}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
            type=self.type,
        )

    __repr__ = __str__

class BuiltinTypeSymbol(Symbol):
    def __init__(self, name):
        super().__init__(name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return "<{class_name}(name='{name}')>".format(
            class_name=self.__class__.__name__,
            name=self.name,
        )

class ProcedureSymbol(Symbol):
    def __init__(self, name, formal_params=None):
        super().__init__(name)
        self.formal_params = [] if formal_params is None else formal_params
        self.block_ast = None

    def __str__(self):
        return '<{class_name}(name={name}, parameters={params})>'.format(
            class_name=self.__class__.__name__,
            name=self.name,
            params=self.formal_params,
        )

    __repr__ = __str__
    
class ScopedSymbolTable:
    def __init__(self, scope_name, scope_level, enclosing_scope=None):
        self._symbols = {}
        self.scope_name = scope_name
        self.scope_level = scope_level
        self.enclosing_scope = enclosing_scope

    def _init_builtins(self):
        self.insert(BuiltinTypeSymbol('INTEGER'))
        self.insert(BuiltinTypeSymbol('REAL'))
        self.insert(BuiltinTypeSymbol('BOOLEAN'))

    def __str__(self):
        h1 = 'SCOPE (SCOPED SYMBOL TABLE)'
        lines = ['\n', h1, '=' * len(h1)]
        for header_name, header_value in (
            ('Scope name', self.scope_name),
            ('Scope level', self.scope_level),
            ('Enclosing scope',
             self.enclosing_scope.scope_name if self.enclosing_scope else None
            )
        ):
            lines.append(f'{header_name:<15}: {header_value}')
        h2 = 'Scope (Scoped symbol table) contents'
        lines.extend([h2, '-' * len(h2)])
        lines.extend(
            f'{key:>7}: {value}'
            for key, value in self._symbols.items()
        )
        lines.append('\n')
        s = '\n'.join(lines)
        return s

    __repr__ = __str__

    def log(self, msg):
        if _SHOULD_LOG_SCOPE:
            print(msg)

    def insert(self, symbol):
        self.log(f'Insert: {symbol.name}')
        symbol.scope_level = self.scope_level
        self._symbols[symbol.name] = symbol

    def lookup(self, name, current_scope_only=False):
        self.log(f'Lookup: {name}. (Scope name: {self.scope_name})')
        symbol = self._symbols.get(name)

        if symbol is not None:
            return symbol

        if current_scope_only:
            return None

        if self.enclosing_scope is not None:
            return self.enclosing_scope.lookup(name)

class SemanticAnalyzer(NodeVisitor):
    def __init__(self):
        self.current_scope = None

    def log(self, msg):
        if _SHOULD_LOG_SCOPE:
            print(msg)

    def error(self, error_code, token):
        raise SemanticError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}',
        )

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_Program(self, node):
        self.log('ENTER scope: global')
        global_scope = ScopedSymbolTable(
            scope_name='global',
            scope_level=1,
            enclosing_scope=self.current_scope, 
        )
        global_scope._init_builtins()
        self.current_scope = global_scope

        self.visit(node.block)

        self.log(global_scope)

        self.current_scope = self.current_scope.enclosing_scope
        self.log('LEAVE scope: global')

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_NoOp(self, node):
        pass

    def visit_IfStmt(self, node):
        self.visit(node.condition)
        for statement in node.consequences:
            self.visit(statement)

        for statement in node.alternatives:
            self.visit(statement)

    def visit_String(self, node):
        pass

    def visit_WriteStmt(self, node):
        for expression in node.expressions:
            self.visit(expression)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_ProcedureDecl(self, node):
        proc_name = node.proc_name
        proc_symbol = ProcedureSymbol(proc_name)
        self.current_scope.insert(proc_symbol)

        self.log(f'ENTER scope: {proc_name}')
        procedure_scope = ScopedSymbolTable(
            scope_name=proc_name,
            scope_level=self.current_scope.scope_level + 1,
            enclosing_scope=self.current_scope
        )
        self.current_scope = procedure_scope

        for param in node.formal_params:
            param_type = self.current_scope.lookup(param.type_node.value)
            param_name = param.var_node.value
            var_symbol = VarSymbol(param_name, param_type)
            self.current_scope.insert(var_symbol)
            proc_symbol.formal_params.append(var_symbol)

        self.visit(node.block_node)

        self.log(procedure_scope)

        self.current_scope = self.current_scope.enclosing_scope
        self.log(f'LEAVE scope: {proc_name}')

        proc_symbol.block_ast = node.block_node

    def visit_VarDecl(self, node):
        type_name = node.type_node.value
        type_symbol = self.current_scope.lookup(type_name)

        var_name = node.var_node.value
        var_symbol = VarSymbol(var_name, type_symbol)

        if self.current_scope.lookup(var_name, current_scope_only=True):
            self.error(
                error_code=ErrorCode.DUPLICATE_ID,
                token=node.var_node.token,
            )

        self.current_scope.insert(var_symbol)

    def visit_Assign(self, node):
        self.visit(node.right)
        self.visit(node.left)

    def visit_Var(self, node):
        var_name = node.value
        var_symbol = self.current_scope.lookup(var_name)
        if var_symbol is None:
            self.error(error_code=ErrorCode.ID_NOT_FOUND, token=node.token)

    def visit_Num(self, node):
        pass

    def visit_Boolean(self, node):
        pass

    def visit_UnaryOp(self, node):
        pass

    def visit_ProcedureCall(self, node):
        for param_node in node.actual_params:
            self.visit(param_node)

        proc_symbol = self.current_scope.lookup(node.proc_name)
        node.proc_symbol = proc_symbol
#  INTERPRETER                                                          
class CallStack:
    def __init__(self):
        self._records = []

    def push(self, ar):
        self._records.append(ar)

    def pop(self):
        return self._records.pop()

    def peek(self):
        return self._records[-1]

    def __str__(self):
        s = '\n'.join(repr(ar) for ar in reversed(self._records))
        s = f'CALL STACK\n{s}\n\n'
        return s

    def __repr__(self):
        return self.__str__()

class ARType(Enum):
    PROGRAM   = 'PROGRAM'
    PROCEDURE = 'PROCEDURE'

class ActivationRecord:
    def __init__(self, name, type, nesting_level):
        self.name = name
        self.type = type
        self.nesting_level = nesting_level
        self.members = {}

    def __setitem__(self, key, value):
        self.members[key] = value

    def __getitem__(self, key):
        return self.members[key]

    def get(self, key):
        return self.members.get(key)

    def __str__(self):
        lines = [
            '{level}: {type} {name}'.format(
                level=self.nesting_level,
                type=self.type.value,
                name=self.name,
            )
        ]
        for name, val in self.members.items():
            lines.append(f'   {name:<20}: {val}')

        s = '\n'.join(lines)
        return s

    def __repr__(self):
        return self.__str__()

class Interpreter(NodeVisitor):
    def __init__(self, tree):
        self.tree = tree
        self.call_stack = CallStack()

    def log(self, msg):
        if _SHOULD_LOG_STACK:
            print(msg)

    def visit_Program(self, node):
        program_name = node.name
        self.log(f'ENTER: PROGRAM {program_name}')

        ar = ActivationRecord(
            name=program_name,
            type=ARType.PROGRAM,
            nesting_level=1,
        )
        self.call_stack.push(ar)

        self.log(str(self.call_stack))

        self.visit(node.block)

        self.log(f'LEAVE: PROGRAM {program_name}')
        self.log(str(self.call_stack))

        self.call_stack.pop()

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_VarDecl(self, node):
        pass

    def visit_Type(self, node):
        pass

    def visit_BinOp(self, node):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.INTEGER_DIV:
            return self.visit(node.left) // self.visit(node.right)
        elif node.op.type == TokenType.FLOAT_DIV:
            return float(self.visit(node.left)) / float(self.visit(node.right))
        elif node.op.type == TokenType.LESS_THAN:
            return self.visit(node.left) < self.visit(node.right)
        elif node.op.type == TokenType.GREATER_THAN:
            return self.visit(node.left) > self.visit(node.right)
        elif node.op.type == TokenType.EQUAL:
            return self.visit(node.left) == self.visit(node.right)
        elif node.op.type == TokenType.LESS_EQUAL:
            return self.visit(node.left) <= self.visit(node.right)
        elif node.op.type == TokenType.GREATER_EQUAL:
            return self.visit(node.left) >= self.visit(node.right)
        elif node.op.type == TokenType.NOT_EQUAL:
            return self.visit(node.left) != self.visit(node.right)

    def visit_Num(self, node):
        return node.value

    def visit_Boolean(self, node):
        return node.value

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == TokenType.PLUS:
            return +self.visit(node.expr)
        elif op == TokenType.MINUS:
            return -self.visit(node.expr)

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_Assign(self, node):
        var_name = node.left.value
        var_value = self.visit(node.right)

        ar = self.call_stack.peek()
        ar[var_name] = var_value

    def visit_Var(self, node):
        var_name = node.value

        ar = self.call_stack.peek()
        var_value = ar.get(var_name)

        return var_value

    def visit_NoOp(self, node):
        pass

    def visit_IfStmt(self, node):
        if self.visit(node.condition):
            for statement in node.consequences:
                self.visit(statement)
        else:
            for statement in node.alternatives:
                self.visit(statement)

    def visit_String(self, node):
        return node.value

    def visit_WriteStmt(self, node):
        if len(node.expressions) > 0:
            for expression in node.expressions:
                value = self.visit(expression)
                print(value, end=" ")
            if node.new_line:
                print() 
        else:
            print(" ", end=" ")
            if node.new_line:                
                print() 

    def visit_ProcedureDecl(self, node):
        pass

    def visit_ProcedureCall(self, node):
        proc_name = node.proc_name
        proc_symbol = node.proc_symbol

        ar = ActivationRecord(
            name=proc_name,
            type=ARType.PROCEDURE,
            nesting_level=proc_symbol.scope_level + 1,
        )

        formal_params = proc_symbol.formal_params
        actual_params = node.actual_params

        for param_symbol, argument_node in zip(formal_params, actual_params):
            ar[param_symbol.name] = self.visit(argument_node)

        self.call_stack.push(ar)

        self.log(f'ENTER: PROCEDURE {proc_name}')
        self.log(str(self.call_stack))

        self.visit(proc_symbol.block_ast)

        self.log(f'LEAVE: PROCEDURE {proc_name}')
        self.log(str(self.call_stack))

        self.call_stack.pop()

    def interpret(self):
        tree = self.tree
        if tree is None:
            return ''
        return self.visit(tree)

def main():
    text = """
    Abjohyahih Main;
        var a, b : inthye;
    shuruyahase
        a := 12;
        agarhomie a = 1 fir
          shuruyahase
            likhline('this is the if 1...');
          khatamhogyatatabyebye
        varna agarhomie a = 2 fir
          shuruyahase
            likhline('this is the if 2...');
          khatamhogyatatabyebye
        varna agarhomie a = 3 fir
          shuruyahase
            likhline('this is the if 3...');
          khatamhogyatatabyebye
        varna agarhomie a = 4 fir
          shuruyahase
            likhline('this is the if 4...');
          khatamhogyatatabyebye
        varna agarhomie a = 5 fir
          shuruyahase
            likhline('this is the if 5...');
          khatamhogyatatabyebye
        varna agarhomie a = 6 fir
          shuruyahase
            likhline('this is the if 6...');
          khatamhogyatatabyebye
        varna agarhomie a = 7 fir
          shuruyahase
            likhline('this is the if 7...');
          khatamhogyatatabyebye
        varna
            agarhomie a = 12 fir
                likho('It is 12! Yeah baby...!!! BTW: this is a nested If... pretty cool eh...?')
            varna
                likho('I dont know what is it :(');
    khatamhogyatatabyebye.
    """
    lexer = Lexer(text)
    try:
            parser = Parser(lexer)
            tree = parser.parse()
    except (LexerError, ParserError) as e:
            print(e.message)
            sys.exit(1)

    semantic_analyzer = SemanticAnalyzer()
    try:
        semantic_analyzer.visit(tree)
    except SemanticError as e:
            print(e.message)
            sys.exit(1)

    interpreter = Interpreter(tree)
    interpreter.interpret()



if __name__ == '__main__':
    main()

