"""
Quantum Domain-Specific Language (DSL) for Coratrix.

This module defines a high-level quantum programming language that can be
compiled to Coratrix IR and then to various target formats.
"""

import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import ast


class DSLToken(Enum):
    """Tokens for the quantum DSL."""
    # Keywords
    QUANTUM = "quantum"
    GATE = "gate"
    CIRCUIT = "circuit"
    MEASURE = "measure"
    IF = "if"
    FOR = "for"
    WHILE = "while"
    DEF = "def"
    RETURN = "return"
    
    # Gates
    H = "h"
    X = "x"
    Y = "y"
    Z = "z"
    CNOT = "cnot"
    CZ = "cz"
    CPHASE = "cphase"
    ROTATION = "rotation"
    CUSTOM = "custom"
    
    # Operators
    ASSIGN = "="
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    POWER = "**"
    
    # Delimiters
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    LBRACE = "{"
    RBRACE = "}"
    COMMA = ","
    SEMICOLON = ";"
    COLON = ":"
    
    # Literals
    NUMBER = "number"
    IDENTIFIER = "identifier"
    STRING = "string"
    
    # Special
    NEWLINE = "newline"
    EOF = "eof"


@dataclass
class Token:
    """A token in the quantum DSL."""
    type: DSLToken
    value: str
    line: int
    column: int


class DSLTokenizer:
    """Tokenizer for the quantum DSL."""
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
    
    def tokenize(self) -> List[Token]:
        """Tokenize the source code."""
        while self.position < len(self.source):
            self._skip_whitespace()
            if self.position >= len(self.source):
                break
            
            char = self.source[self.position]
            
            if char.isalpha() or char == '_':
                self._tokenize_identifier_or_keyword()
            elif char.isdigit() or char == '.':
                self._tokenize_number()
            elif char == '"' or char == "'":
                self._tokenize_string()
            elif char in '()[]{}':
                self._tokenize_delimiter()
            elif char in '+-*/=<>!':
                self._tokenize_operator()
            elif char == ';':
                self._add_token(DSLToken.SEMICOLON, ';')
                self._advance()
            elif char == ',':
                self._add_token(DSLToken.COMMA, ',')
                self._advance()
            elif char == ':':
                self._add_token(DSLToken.COLON, ':')
                self._advance()
            else:
                self._advance()  # Skip unknown characters
        
        self._add_token(DSLToken.EOF, '')
        return self.tokens
    
    def _skip_whitespace(self):
        """Skip whitespace and comments."""
        while self.position < len(self.source):
            char = self.source[self.position]
            if char.isspace():
                if char == '\n':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.position += 1
            elif char == '#':
                # Skip line comment
                while self.position < len(self.source) and self.source[self.position] != '\n':
                    self.position += 1
            else:
                break
    
    def _tokenize_identifier_or_keyword(self):
        """Tokenize an identifier or keyword."""
        start = self.position
        while (self.position < len(self.source) and 
               (self.source[self.position].isalnum() or self.source[self.position] == '_')):
            self.position += 1
        
        value = self.source[start:self.position]
        
        # Check if it's a keyword
        keyword_map = {
            'quantum': DSLToken.QUANTUM,
            'gate': DSLToken.GATE,
            'circuit': DSLToken.CIRCUIT,
            'measure': DSLToken.MEASURE,
            'if': DSLToken.IF,
            'for': DSLToken.FOR,
            'while': DSLToken.WHILE,
            'def': DSLToken.DEF,
            'return': DSLToken.RETURN,
            'h': DSLToken.H,
            'x': DSLToken.X,
            'y': DSLToken.Y,
            'z': DSLToken.Z,
            'cnot': DSLToken.CNOT,
            'cz': DSLToken.CZ,
            'cphase': DSLToken.CPHASE,
            'rotation': DSLToken.ROTATION,
            'custom': DSLToken.CUSTOM
        }
        
        token_type = keyword_map.get(value.lower(), DSLToken.IDENTIFIER)
        self._add_token(token_type, value)
    
    def _tokenize_number(self):
        """Tokenize a number."""
        start = self.position
        while (self.position < len(self.source) and 
               (self.source[self.position].isdigit() or self.source[self.position] == '.')):
            self.position += 1
        
        value = self.source[start:self.position]
        self._add_token(DSLToken.NUMBER, value)
    
    def _tokenize_string(self):
        """Tokenize a string literal."""
        quote = self.source[self.position]
        self.position += 1
        start = self.position
        
        while self.position < len(self.source) and self.source[self.position] != quote:
            self.position += 1
        
        if self.position < len(self.source):
            value = self.source[start:self.position]
            self.position += 1
            self._add_token(DSLToken.STRING, value)
    
    def _tokenize_delimiter(self):
        """Tokenize a delimiter."""
        char = self.source[self.position]
        delimiter_map = {
            '(': DSLToken.LPAREN,
            ')': DSLToken.RPAREN,
            '[': DSLToken.LBRACKET,
            ']': DSLToken.RBRACKET,
            '{': DSLToken.LBRACE,
            '}': DSLToken.RBRACE
        }
        
        token_type = delimiter_map[char]
        self._add_token(token_type, char)
        self._advance()
    
    def _tokenize_operator(self):
        """Tokenize an operator."""
        char = self.source[self.position]
        
        if char == '=':
            self._add_token(DSLToken.ASSIGN, '=')
        elif char == '+':
            self._add_token(DSLToken.PLUS, '+')
        elif char == '-':
            self._add_token(DSLToken.MINUS, '-')
        elif char == '*':
            if (self.position + 1 < len(self.source) and 
                self.source[self.position + 1] == '*'):
                self._add_token(DSLToken.POWER, '**')
                self.position += 2
                return
            else:
                self._add_token(DSLToken.MULTIPLY, '*')
        elif char == '/':
            self._add_token(DSLToken.DIVIDE, '/')
        
        self._advance()
    
    def _add_token(self, token_type: DSLToken, value: str):
        """Add a token to the list."""
        token = Token(token_type, value, self.line, self.column)
        self.tokens.append(token)
    
    def _advance(self):
        """Advance the position."""
        self.position += 1
        self.column += 1


@dataclass
class DSLNode:
    """Base class for DSL AST nodes."""
    line: int
    column: int


@dataclass
class QuantumProgram(DSLNode):
    """Root node for a quantum program."""
    circuits: List['QuantumCircuit']
    gates: List['GateDefinition']
    functions: List['FunctionDefinition']


@dataclass
class QuantumCircuit(DSLNode):
    """A quantum circuit definition."""
    name: str
    parameters: List[str]
    body: List['Statement']


@dataclass
class GateDefinition(DSLNode):
    """A custom gate definition."""
    name: str
    parameters: List[str]
    qubits: List[str]
    body: List['Statement']


@dataclass
class FunctionDefinition(DSLNode):
    """A function definition."""
    name: str
    parameters: List[str]
    return_type: Optional[str]
    body: List['Statement']


@dataclass
class Statement(DSLNode):
    """Base class for statements."""
    pass


@dataclass
class GateCall(Statement):
    """A gate call statement."""
    gate_name: str
    qubits: List['Expression']
    parameters: List['Expression']


@dataclass
class MeasureStatement(Statement):
    """A measurement statement."""
    qubit: 'Expression'
    classical_bit: 'Expression'


@dataclass
class IfStatement(Statement):
    """An if statement."""
    condition: 'Expression'
    then_body: List[Statement]
    else_body: Optional[List[Statement]]


@dataclass
class ForStatement(Statement):
    """A for loop statement."""
    variable: str
    start: 'Expression'
    end: 'Expression'
    step: Optional['Expression']
    body: List[Statement]


@dataclass
class WhileStatement(Statement):
    """A while loop statement."""
    condition: 'Expression'
    body: List[Statement]


@dataclass
class Expression(DSLNode):
    """Base class for expressions."""
    pass


@dataclass
class NumberLiteral(Expression):
    """A number literal."""
    value: Union[int, float]


@dataclass
class StringLiteral(Expression):
    """A string literal."""
    value: str


@dataclass
class Identifier(Expression):
    """An identifier."""
    name: str


@dataclass
class BinaryExpression(Expression):
    """A binary expression."""
    left: Expression
    operator: str
    right: Expression


@dataclass
class UnaryExpression(Expression):
    """A unary expression."""
    operator: str
    operand: Expression


@dataclass
class FunctionCall(Expression):
    """A function call expression."""
    name: str
    arguments: List[Expression]


class DSLParser:
    """Parser for the quantum DSL."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
    
    def parse(self) -> QuantumProgram:
        """Parse the tokens into an AST."""
        circuits = []
        gates = []
        functions = []
        
        while not self._is_at_end():
            if self._match(DSLToken.CIRCUIT):
                circuits.append(self._parse_circuit())
            elif self._match(DSLToken.GATE):
                gates.append(self._parse_gate_definition())
            elif self._match(DSLToken.DEF):
                functions.append(self._parse_function_definition())
            else:
                self._advance()
        
        return QuantumProgram(0, 0, circuits, gates, functions)
    
    def _parse_circuit(self) -> QuantumCircuit:
        """Parse a circuit definition."""
        name = self._consume(DSLToken.IDENTIFIER, "Expected circuit name").value
        
        # Parse parameters
        parameters = []
        if self._match(DSLToken.LPAREN):
            if not self._match(DSLToken.RPAREN):
                while True:
                    parameters.append(self._consume(DSLToken.IDENTIFIER, "Expected parameter name").value)
                    if not self._match(DSLToken.COMMA):
                        break
                self._consume(DSLToken.RPAREN, "Expected ')'")
        
        # Parse body
        self._consume(DSLToken.LBRACE, "Expected '{'")
        body = self._parse_statement_list()
        self._consume(DSLToken.RBRACE, "Expected '}'")
        
        return QuantumCircuit(0, 0, name, parameters, body)
    
    def _parse_gate_definition(self) -> GateDefinition:
        """Parse a gate definition."""
        name = self._consume(DSLToken.IDENTIFIER, "Expected gate name").value
        
        # Parse parameters
        parameters = []
        if self._match(DSLToken.LPAREN):
            if not self._match(DSLToken.RPAREN):
                while True:
                    parameters.append(self._consume(DSLToken.IDENTIFIER, "Expected parameter name").value)
                    if not self._match(DSLToken.COMMA):
                        break
                self._consume(DSLToken.RPAREN, "Expected ')'")
        
        # Parse qubits
        qubits = []
        self._consume(DSLToken.LPAREN, "Expected '(' for qubits")
        if not self._match(DSLToken.RPAREN):
            while True:
                qubits.append(self._consume(DSLToken.IDENTIFIER, "Expected qubit name").value)
                if not self._match(DSLToken.COMMA):
                    break
            self._consume(DSLToken.RPAREN, "Expected ')'")
        
        # Parse body
        self._consume(DSLToken.LBRACE, "Expected '{'")
        body = self._parse_statement_list()
        self._consume(DSLToken.RBRACE, "Expected '}'")
        
        return GateDefinition(0, 0, name, parameters, qubits, body)
    
    def _parse_function_definition(self) -> FunctionDefinition:
        """Parse a function definition."""
        name = self._consume(DSLToken.IDENTIFIER, "Expected function name").value
        
        # Parse parameters
        parameters = []
        self._consume(DSLToken.LPAREN, "Expected '('")
        if not self._match(DSLToken.RPAREN):
            while True:
                parameters.append(self._consume(DSLToken.IDENTIFIER, "Expected parameter name").value)
                if not self._match(DSLToken.COMMA):
                    break
            self._consume(DSLToken.RPAREN, "Expected ')'")
        
        # Parse return type (optional)
        return_type = None
        if self._match(DSLToken.COLON):
            return_type = self._consume(DSLToken.IDENTIFIER, "Expected return type").value
        
        # Parse body
        self._consume(DSLToken.LBRACE, "Expected '{'")
        body = self._parse_statement_list()
        self._consume(DSLToken.RBRACE, "Expected '}'")
        
        return FunctionDefinition(0, 0, name, parameters, return_type, body)
    
    def _parse_statement_list(self) -> List[Statement]:
        """Parse a list of statements."""
        statements = []
        while not self._is_at_end() and not self._check(DSLToken.RBRACE):
            statements.append(self._parse_statement())
        return statements
    
    def _parse_statement(self) -> Statement:
        """Parse a single statement."""
        if self._match(DSLToken.MEASURE):
            return self._parse_measure_statement()
        elif self._match(DSLToken.IF):
            return self._parse_if_statement()
        elif self._match(DSLToken.FOR):
            return self._parse_for_statement()
        elif self._match(DSLToken.WHILE):
            return self._parse_while_statement()
        else:
            # Try to parse as gate call
            return self._parse_gate_call()
    
    def _parse_measure_statement(self) -> MeasureStatement:
        """Parse a measure statement."""
        qubit = self._parse_expression()
        self._consume(DSLToken.ASSIGN, "Expected '='")
        classical_bit = self._parse_expression()
        self._consume(DSLToken.SEMICOLON, "Expected ';'")
        return MeasureStatement(0, 0, qubit, classical_bit)
    
    def _parse_if_statement(self) -> IfStatement:
        """Parse an if statement."""
        self._consume(DSLToken.LPAREN, "Expected '('")
        condition = self._parse_expression()
        self._consume(DSLToken.RPAREN, "Expected ')'")
        
        self._consume(DSLToken.LBRACE, "Expected '{'")
        then_body = self._parse_statement_list()
        self._consume(DSLToken.RBRACE, "Expected '}'")
        
        else_body = None
        if self._match(DSLToken.ELSE):
            self._consume(DSLToken.LBRACE, "Expected '{'")
            else_body = self._parse_statement_list()
            self._consume(DSLToken.RBRACE, "Expected '}'")
        
        return IfStatement(0, 0, condition, then_body, else_body)
    
    def _parse_for_statement(self) -> ForStatement:
        """Parse a for statement."""
        variable = self._consume(DSLToken.IDENTIFIER, "Expected variable name").value
        self._consume(DSLToken.ASSIGN, "Expected '='")
        start = self._parse_expression()
        self._consume(DSLToken.COMMA, "Expected ','")
        end = self._parse_expression()
        
        step = None
        if self._match(DSLToken.COMMA):
            step = self._parse_expression()
        
        self._consume(DSLToken.RPAREN, "Expected ')'")
        
        self._consume(DSLToken.LBRACE, "Expected '{'")
        body = self._parse_statement_list()
        self._consume(DSLToken.RBRACE, "Expected '}'")
        
        return ForStatement(0, 0, variable, start, end, step, body)
    
    def _parse_while_statement(self) -> WhileStatement:
        """Parse a while statement."""
        self._consume(DSLToken.LPAREN, "Expected '('")
        condition = self._parse_expression()
        self._consume(DSLToken.RPAREN, "Expected ')'")
        
        self._consume(DSLToken.LBRACE, "Expected '{'")
        body = self._parse_statement_list()
        self._consume(DSLToken.RBRACE, "Expected '}'")
        
        return WhileStatement(0, 0, condition, body)
    
    def _parse_gate_call(self) -> GateCall:
        """Parse a gate call."""
        gate_name = self._consume(DSLToken.IDENTIFIER, "Expected gate name").value
        
        # Parse qubits
        qubits = []
        if self._match(DSLToken.LPAREN):
            if not self._match(DSLToken.RPAREN):
                while True:
                    qubits.append(self._parse_expression())
                    if not self._match(DSLToken.COMMA):
                        break
                self._consume(DSLToken.RPAREN, "Expected ')'")
        
        # Parse parameters (optional)
        parameters = []
        if self._match(DSLToken.LBRACKET):
            if not self._match(DSLToken.RBRACKET):
                while True:
                    parameters.append(self._parse_expression())
                    if not self._match(DSLToken.COMMA):
                        break
                self._consume(DSLToken.RBRACKET, "Expected ']'")
        
        self._consume(DSLToken.SEMICOLON, "Expected ';'")
        return GateCall(0, 0, gate_name, qubits, parameters)
    
    def _parse_expression(self) -> Expression:
        """Parse an expression."""
        return self._parse_assignment()
    
    def _parse_assignment(self) -> Expression:
        """Parse an assignment expression."""
        expr = self._parse_equality()
        
        if self._match(DSLToken.ASSIGN):
            right = self._parse_assignment()
            return BinaryExpression(0, 0, expr, "=", right)
        
        return expr
    
    def _parse_equality(self) -> Expression:
        """Parse an equality expression."""
        expr = self._parse_comparison()
        
        while self._match(DSLToken.EQUAL, DSLToken.NOT_EQUAL):
            operator = self._previous().value
            right = self._parse_comparison()
            expr = BinaryExpression(0, 0, expr, operator, right)
        
        return expr
    
    def _parse_comparison(self) -> Expression:
        """Parse a comparison expression."""
        expr = self._parse_term()
        
        while self._match(DSLToken.GREATER, DSLToken.GREATER_EQUAL, 
                         DSLToken.LESS, DSLToken.LESS_EQUAL):
            operator = self._previous().value
            right = self._parse_term()
            expr = BinaryExpression(0, 0, expr, operator, right)
        
        return expr
    
    def _parse_term(self) -> Expression:
        """Parse a term expression."""
        expr = self._parse_factor()
        
        while self._match(DSLToken.PLUS, DSLToken.MINUS):
            operator = self._previous().value
            right = self._parse_factor()
            expr = BinaryExpression(0, 0, expr, operator, right)
        
        return expr
    
    def _parse_factor(self) -> Expression:
        """Parse a factor expression."""
        expr = self._parse_unary()
        
        while self._match(DSLToken.MULTIPLY, DSLToken.DIVIDE):
            operator = self._previous().value
            right = self._parse_unary()
            expr = BinaryExpression(0, 0, expr, operator, right)
        
        return expr
    
    def _parse_unary(self) -> Expression:
        """Parse a unary expression."""
        if self._match(DSLToken.MINUS, DSLToken.NOT):
            operator = self._previous().value
            right = self._parse_unary()
            return UnaryExpression(0, 0, operator, right)
        
        return self._parse_primary()
    
    def _parse_primary(self) -> Expression:
        """Parse a primary expression."""
        if self._match(DSLToken.NUMBER):
            value = float(self._previous().value)
            return NumberLiteral(0, 0, value)
        
        if self._match(DSLToken.STRING):
            value = self._previous().value
            return StringLiteral(0, 0, value)
        
        if self._match(DSLToken.IDENTIFIER):
            name = self._previous().value
            
            # Check if it's a function call
            if self._match(DSLToken.LPAREN):
                arguments = []
                if not self._match(DSLToken.RPAREN):
                    while True:
                        arguments.append(self._parse_expression())
                        if not self._match(DSLToken.COMMA):
                            break
                    self._consume(DSLToken.RPAREN, "Expected ')'")
                return FunctionCall(0, 0, name, arguments)
            
            return Identifier(0, 0, name)
        
        if self._match(DSLToken.LPAREN):
            expr = self._parse_expression()
            self._consume(DSLToken.RPAREN, "Expected ')'")
            return expr
        
        raise self._error("Expected expression")
    
    def _match(self, *token_types: DSLToken) -> bool:
        """Check if current token matches any of the given types."""
        for token_type in token_types:
            if self._check(token_type):
                self._advance()
                return True
        return False
    
    def _check(self, token_type: DSLToken) -> bool:
        """Check if current token is of the given type."""
        if self._is_at_end():
            return False
        return self._peek().type == token_type
    
    def _advance(self) -> Token:
        """Advance to the next token."""
        if not self._is_at_end():
            self.position += 1
        return self._previous()
    
    def _is_at_end(self) -> bool:
        """Check if we're at the end of tokens."""
        return self._peek().type == DSLToken.EOF
    
    def _peek(self) -> Token:
        """Peek at the current token."""
        return self.tokens[self.position]
    
    def _previous(self) -> Token:
        """Get the previous token."""
        return self.tokens[self.position - 1]
    
    def _consume(self, token_type: DSLToken, message: str) -> Token:
        """Consume a token of the expected type."""
        if self._check(token_type):
            return self._advance()
        
        raise self._error(message)
    
    def _error(self, message: str) -> Exception:
        """Create an error with the current token."""
        token = self._peek()
        return Exception(f"Line {token.line}, Column {token.column}: {message}")


class QuantumDSL:
    """High-level interface for the quantum DSL."""
    
    def __init__(self):
        self.tokenizer = None
        self.parser = None
    
    def compile(self, source: str) -> 'QuantumProgram':
        """Compile DSL source to AST."""
        self.tokenizer = DSLTokenizer(source)
        tokens = self.tokenizer.tokenize()
        
        self.parser = DSLParser(tokens)
        return self.parser.parse()
    
    def parse_file(self, filename: str) -> 'QuantumProgram':
        """Parse a DSL file."""
        with open(filename, 'r') as f:
            source = f.read()
        return self.compile(source)
