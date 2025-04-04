"""Test script for list operations."""

import re
from typing import Any, Optional, Tuple

class CalculatorInterface:
    """Interface for all calculators."""
    
    def parse(self, query: str) -> Tuple[Optional[Any], Optional[str]]:
        """Parse a query and return result or error."""
        raise NotImplementedError()

class ListCalculator(CalculatorInterface):
    """Calculator for list operations.
    
    Supports list creation and operations with scalars or other lists.
    """
    
    # Pattern for list operations: "list + scalar", "list * list", etc.
    # Captures: list_var, operator, operand (number or variable)
    list_operation_pattern = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([\+\-\*\/])\s*([\d\.]+|[a-zA-Z_][a-zA-Z0-9_]*)\s*$")
    
    def __init__(self):
        pass
        
    def parse(self, query: str) -> Tuple[Optional[Any], Optional[str]]:
        """Parse list operations.
        
        Examples:
        - "x * 10" (where x is a list)
        - "x + y" (where x and y are lists)
        """
        # This parse method will be called with variable-substituted values
        # So we need to check if the input might be a list operation after substitution
        
        # For direct list operation expressions (with variables)
        match = self.list_operation_pattern.match(query)
        if match:
            return {"type": "list_operation", "pattern": match.groups()}, None
            
        # Check if the query is a list literal (after substitution)
        if query.startswith('[') and query.endswith(']'):
            try:
                # Parse the list
                list_str = query.strip()
                # Simple parsing, assumes well-formed list
                items_str = list_str[1:-1].split(',')
                items = [float(item.strip()) for item in items_str]
                
                # Convert integers to int for cleaner output
                items = [int(item) if item == int(item) else item for item in items]
                
                return items, None
            except ValueError:
                return None, "Invalid number in list"
            except Exception as e:
                return None, f"Error parsing list: {e}"
        
        # Check if this is a list operation expression (after substitution)
        # This would be like "[1, 2, 3] * 2" after variables are substituted
        list_op_match = re.match(r"^\s*(\[.+\])\s*([\+\-\*\/])\s*(.+)$", query)
        if list_op_match:
            try:
                list_str, operator, operand_str = list_op_match.groups()
                
                # Parse the list
                list_str = list_str.strip()
                items_str = list_str[1:-1].split(',')
                items = [float(item.strip()) for item in items_str]
                
                # Convert integers to int for cleaner output
                items = [int(item) if item == int(item) else item for item in items]
                
                # Parse the operand - could be a scalar or another list
                if operand_str.strip().startswith('[') and operand_str.strip().endswith(']'):
                    # Operand is a list
                    operand_str = operand_str.strip()
                    operand_items_str = operand_str[1:-1].split(',')
                    operand = [float(item.strip()) for item in operand_items_str]
                    # Convert integers to int
                    operand = [int(op) if op == int(op) else op for op in operand]
                else:
                    # Operand is a scalar
                    operand = float(operand_str.strip())
                    if operand == int(operand):
                        operand = int(operand)
                
                # Perform the operation
                result = self._perform_list_operation(items, operator, operand)
                
                # Format the result
                return result, None
            except ValueError:
                return None, "Invalid number in list operation"
            except Exception as e:
                return None, f"Error performing list operation: {e}"
                
        return None, None
    
    def process_list_operation(self, list_var: Any, operator: str, operand: Any) -> list:
        """Process a list operation with a given operator and operand."""
        return self._perform_list_operation(list_var, operator, operand)
        
    def _perform_list_operation(self, list_var: list, operator: str, operand: Any) -> list:
        """Perform arithmetic operation between a list and a scalar or another list."""
        if operator == '+':
            if isinstance(operand, list):
                # List + List: Element-wise addition
                if len(list_var) != len(operand):
                    raise ValueError(f"Lists must have the same length for addition: {len(list_var)} != {len(operand)}")
                return [a + b for a, b in zip(list_var, operand)]
            else:
                # List + Scalar: Add scalar to each element
                return [item + operand for item in list_var]
                
        elif operator == '-':
            if isinstance(operand, list):
                # List - List: Element-wise subtraction
                if len(list_var) != len(operand):
                    raise ValueError(f"Lists must have the same length for subtraction: {len(list_var)} != {len(operand)}")
                return [a - b for a, b in zip(list_var, operand)]
            else:
                # List - Scalar: Subtract scalar from each element
                return [item - operand for item in list_var]
                
        elif operator == '*':
            if isinstance(operand, list):
                # List * List: Element-wise multiplication
                if len(list_var) != len(operand):
                    raise ValueError(f"Lists must have the same length for multiplication: {len(list_var)} != {len(operand)}")
                return [a * b for a, b in zip(list_var, operand)]
            else:
                # List * Scalar: Multiply each element by scalar
                return [item * operand for item in list_var]
                
        elif operator == '/':
            if isinstance(operand, list):
                # List / List: Element-wise division
                if len(list_var) != len(operand):
                    raise ValueError(f"Lists must have the same length for division: {len(list_var)} != {len(operand)}")
                if 0 in operand:
                    raise ZeroDivisionError("Division by zero in list operation")
                return [a / b for a, b in zip(list_var, operand)]
            else:
                # List / Scalar: Divide each element by scalar
                if operand == 0:
                    raise ZeroDivisionError("Division by zero")
                return [item / operand for item in list_var]
        
        raise ValueError(f"Unsupported operator for list operation: {operator}")

def test_list_assignment():
    """Test list assignment and scalar operations."""
    # Simulating x = [1, 2, 3]
    variables = {}
    list_str = "[1, 2, 3]"
    list_match = re.match(r"^\s*\[(.+)\]\s*$", list_str)
    if list_match:
        list_items_str = list_match.group(1).split(',')
        list_items = [float(item.strip()) for item in list_items_str]
        list_items = [int(item) if item == int(item) else item for item in list_items]
        variables["x"] = list_items
    
    print(f"x = {variables['x']}")
    
    # Test list operations
    calculator = ListCalculator()
    
    # Test list * scalar: x * 10
    operation = "x * 10"
    match = calculator.list_operation_pattern.match(operation)
    if match:
        var_name, operator, operand_str = match.groups()
        if var_name in variables:
            operand = float(operand_str)
            if operand == int(operand):
                operand = int(operand)
            result = calculator.process_list_operation(variables[var_name], operator, operand)
            print(f"{var_name} {operator} {operand_str} = {result}")
    
    # Add another list: y = [4, 5, 6]
    variables["y"] = [4, 5, 6]
    print(f"y = {variables['y']}")
    
    # Test list + list: x + y
    operation = "x + y"
    match = calculator.list_operation_pattern.match(operation)
    if match:
        var_name, operator, operand_str = match.groups()
        if var_name in variables and operand_str in variables:
            result = calculator.process_list_operation(variables[var_name], operator, variables[operand_str])
            print(f"{var_name} {operator} {operand_str} = {result}")
    
    # Test list * list: x * y
    operation = "x * y"
    match = calculator.list_operation_pattern.match(operation)
    if match:
        var_name, operator, operand_str = match.groups()
        if var_name in variables and operand_str in variables:
            result = calculator.process_list_operation(variables[var_name], operator, variables[operand_str])
            print(f"{var_name} {operator} {operand_str} = {result}")

if __name__ == "__main__":
    test_list_assignment()