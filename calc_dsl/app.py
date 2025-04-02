import flask
from flask import request, jsonify
import re
import datetime
import math
from pint import UnitRegistry, UndefinedUnitError
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Dict
import uuid
import sqlite3
import json
import os
import logging

# --- Configuration ---
DATABASE = "sessions.db"
DEBUG_MODE = True  # Set to False in production

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Database Setup ---


def get_db():
    """Connects to the specific database."""
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row  # Access columns by name
    return db


def init_db():
    """Initializes the database schema."""
    if os.path.exists(DATABASE):
        logger.info("Database already exists.")
        # You might want to add migration logic here in a real application
        # For this example, we assume the schema is correct if the file exists
        return

    logger.info(f"Initializing database: {DATABASE}")
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                variables TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        db.commit()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise  # Re-raise the exception to halt startup if DB fails
    finally:
        if db:
            db.close()


# --- Session Persistence Functions ---


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Loads session variables from the database."""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            "SELECT variables FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row["variables"])
        else:
            return None  # Session not found
    except sqlite3.Error as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        return None  # Indicate error or non-existence
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse variables for session {session_id}: {e}")
        return (
            {}
        )  # Return empty dict if data is corrupted? Or None? Let's return empty.
    finally:
        if db:
            db.close()


def save_session(session_id: str, variables: Dict[str, Any]):
    """Saves or updates session variables in the database."""
    try:
        db = get_db()
        cursor = db.cursor()
        variables_json = json.dumps(variables)
        # Use INSERT OR REPLACE (UPSERT)
        cursor.execute(
            """
            INSERT OR REPLACE INTO sessions (session_id, variables, last_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """,
            (session_id, variables_json),
        )
        db.commit()
        logger.debug(f"Session {session_id} saved.")
    except sqlite3.Error as e:
        logger.error(f"Failed to save session {session_id}: {e}")
        # Decide if we should raise an error to the user
    finally:
        if db:
            db.close()


def delete_session_db(session_id: str) -> bool:
    """Deletes a session from the database. Returns True if deleted, False otherwise."""
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        deleted = cursor.rowcount > 0
        db.commit()
        logger.info(f"Session {session_id} deleted: {deleted}")
        return deleted
    except sqlite3.Error as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        return False
    finally:
        if db:
            db.close()


def create_session_db() -> Optional[str]:
    """Creates a new session entry in the database and returns the session ID."""
    session_id = str(uuid.uuid4())
    initial_variables = {}
    try:
        db = get_db()
        cursor = db.cursor()
        variables_json = json.dumps(initial_variables)
        cursor.execute(
            """
            INSERT INTO sessions (session_id, variables)
            VALUES (?, ?)
        """,
            (session_id, variables_json),
        )
        db.commit()
        logger.info(f"New session created: {session_id}")
        return session_id
    except sqlite3.Error as e:
        logger.error(f"Failed to create session: {e}")
        return None
    finally:
        if db:
            db.close()


# --- Variable Substitution ---
# Simple regex for valid variable names (letters, numbers, underscores, not starting with number)
# Using \b for word boundaries is crucial here
VAR_NAME_REGEX = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")


def substitute_variables(expression: str, variables: Dict[str, Any]) -> str:
    """Substitutes known variables into an expression string."""

    # Find all potential variable names
    potential_vars = VAR_NAME_REGEX.findall(expression)

    substituted_expression = expression

    # Sort potential variables by length descending to replace longer names first (e.g., 'va' before 'v')
    # Filter to only include those actually defined in the session variables
    vars_to_replace = sorted(
        [var for var in potential_vars if var in variables], key=len, reverse=True
    )

    for var_name in vars_to_replace:
        value = variables[var_name]
        # Need to handle string vs numeric substitution carefully.
        # Just converting to string might work for math/units, but could be tricky.
        # Let's convert to string representation for replacement.
        # Using word boundaries in the replacement regex helps avoid partial replacements.
        # Ensure the value doesn't contain characters that break the expression (e.g. if value is itself an expression)
        # For now, assume values are simple numbers or results that stringify cleanly.
        try:
            value_str = str(value)
            # Replace var_name only if it appears as a whole word
            substituted_expression = re.sub(
                r"\b" + re.escape(var_name) + r"\b", value_str, substituted_expression
            )
        except Exception as e:
            logger.warning(
                f"Could not substitute variable '{var_name}' with value '{value}': {e}"
            )
            # Decide whether to raise an error or just skip substitution for this var

    logger.debug(
        f"Original expression: '{expression}', Substituted: '{substituted_expression}' with vars: {variables}"
    )
    return substituted_expression


# --- Calculator Interface and Implementations (Mostly Unchanged) ---
# (Keep the CalculatorInterface, TimeCalculator, UnitCalculator, MathCalculator classes as defined in the previous step)
# ... [Calculator classes code from previous answer] ...
class CalculatorInterface(ABC):
    @abstractmethod
    def parse(self, query: str) -> Tuple[Optional[Any], Optional[str]]:
        pass


class TimeCalculator(CalculatorInterface):
    # More flexible time pattern that supports different formats and optional output format
    time_pattern = re.compile(
        r"^\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*-\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:in\s+(\w+(?:-\w+)?))?\s*$",
        re.IGNORECASE,
    )

    def parse(self, query: str) -> Tuple[Optional[Any], Optional[str]]:
        match = self.time_pattern.match(query)
        if not match:
            return None, None

        time_str1 = match.group(1).strip()
        time_str2 = match.group(2).strip()
        output_format = match.group(3).lower() if match.group(3) else "human"

        try:
            # Parse the time strings into datetime objects
            dt1 = self._parse_time_string(time_str1)
            dt2 = self._parse_time_string(time_str2)

            if dt1 is None or dt2 is None:
                return None, "Error parsing time format. Try formats like '9am', '9:30am', '9:30', '13:45'."

            difference = abs(dt1 - dt2)
            total_seconds = int(difference.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            # Format output based on specified format
            if output_format == "human":
                parts = []
                if hours > 0:
                    parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
                if minutes > 0:
                    parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
                if not parts:
                    return "0 minutes", None
                return ", ".join(parts), None
                
            elif output_format == "24-hour":
                return f"{hours:02d}:{minutes:02d}", None
                
            elif output_format == "am-pm":
                period = "AM"
                display_hours = hours
                if hours >= 12:
                    period = "PM"
                    if hours > 12:
                        display_hours = hours - 12
                if display_hours == 0:
                    display_hours = 12
                return f"{display_hours}:{minutes:02d} {period}", None
                
            elif output_format == "seconds":
                return str(total_seconds), None
                
            elif output_format == "minutes":
                total_minutes = (hours * 60) + minutes
                return str(total_minutes), None
                
            elif output_format == "hours":
                total_hours = hours + (minutes / 60)
                return f"{total_hours:.2f}", None
                
            else:
                return None, f"Unknown output format: '{output_format}'. Supported formats: human, 24-hour, am-pm, seconds, minutes, hours."
                
        except Exception as e:
            return None, f"An unexpected error occurred during time calculation: {e}"

    def _parse_time_string(self, time_str: str) -> Optional[datetime.datetime]:
        """Parse a time string in various formats into a datetime object."""
        base_date = datetime.date.today()
        time_str = time_str.strip().lower()

        # Try different time formats
        formats_to_try = []

        # Check if time has AM/PM indicator
        has_ampm = 'am' in time_str or 'pm' in time_str

        # Format the time string for parsing
        if ':' in time_str:  # Has hours and minutes
            if has_ampm:
                # 12-hour format with minutes (e.g., "9:30am", "1:45pm")
                formats_to_try.append((time_str, "%I:%M%p"))
            else:
                # 24-hour format (e.g., "13:45", "9:30")
                formats_to_try.append((time_str, "%H:%M"))
                # Also try 12-hour interpretation if less than 13
                hour = int(time_str.split(':')[0])
                if hour < 13:
                    formats_to_try.append((time_str + "am", "%I:%M%p"))  # Assume AM
        else:  # Just hours, no minutes
            if has_ampm:
                # 12-hour format without minutes (e.g., "9am", "3pm")
                formats_to_try.append((time_str, "%I%p"))
            else:
                # Could be 24-hour format or 12-hour without AM/PM
                hour = int(time_str)
                if hour < 13:
                    # Try as both AM and 24-hour
                    formats_to_try.append((time_str + "am", "%I%p"))  # Assume AM
                    formats_to_try.append((time_str, "%H"))  # 24-hour
                else:
                    # Must be 24-hour
                    formats_to_try.append((time_str, "%H"))

        # Try each format until one works
        for formatted_str, format_str in formats_to_try:
            try:
                time_obj = datetime.datetime.strptime(formatted_str, format_str).time()
                return datetime.datetime.combine(base_date, time_obj)
            except ValueError:
                continue

        return None


class UnitCalculator(CalculatorInterface):
    unit_pattern = re.compile(
        r"^\s*([\d\.\-]+)\s*([\w\s]+?)\s+to\s+([\w\s]+?)\s*$", re.IGNORECASE
    )  # Allow negative numbers

    def __init__(self):
        self.ureg = UnitRegistry()
        self.Q_ = self.ureg.Quantity

    def parse(self, query: str) -> Tuple[Optional[Any], Optional[str]]:
        match = self.unit_pattern.match(query)
        if not match:
            return None, None
        value_str, from_unit, to_unit = (
            match.group(1),
            match.group(2).strip(),
            match.group(3).strip(),
        )
        try:
            value = float(value_str)
            quantity = self.Q_(value, from_unit)
            result_quantity = quantity.to(to_unit)
            # Use a standard format, maybe avoid '~P' if it causes issues with variable reuse
            # Let's use the magnitude and unit name explicitly for better predictability
            # return f"{result_quantity.magnitude} {result_quantity.units}", None
            # Or stick with pretty format for now:
            return f"{result_quantity:~P}", None
        except UndefinedUnitError as e:
            return None, f"Unit Error: {e}"
        except ValueError:
            return None, f"Invalid number '{value_str}' in unit conversion."
        except Exception as e:
            return None, f"An unexpected error during unit conversion: {e}"


class MathCalculator(CalculatorInterface):
    # Traditional math expression pattern with flexible spacing
    traditional_math_pattern = re.compile(r"^\s*(-?[\d\.]+)\s*(%?)\s*([\+\-\*\/\%\^])\s*(-?[\d\.]+)\s*(%?)\s*$")
    
    # Percent operation pattern: "X% of Y" with flexible spacing
    percent_of_pattern = re.compile(r"^\s*(-?[\d\.]+)\s*%\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE)
    
    # Alternative percent pattern: "X percent of Y" (already handled in humanized_patterns but added as a reminder)
    # Alternative percent pattern: "X % of Y" with space before %
    alt_percent_pattern = re.compile(r"^\s*(-?[\d\.]+)\s+%\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE)
    
    # Humanized math patterns for common operations with very flexible spacing
    humanized_patterns = [
        # Power: "X power of Y", "X to the power of Y", "X raised to Y"
        (re.compile(r"^\s*(-?[\d\.]+)\s+(?:to\s+the\s+)?power\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "power"),
        (re.compile(r"^\s*(-?[\d\.]+)\s+raised\s+to\s+(?:the\s+)?(?:power\s+of\s+)?(-?[\d\.]+)\s*$", re.IGNORECASE), "power"),
        (re.compile(r"^\s*(-?[\d\.]+)\s*\^\s*(-?[\d\.]+)\s*$"), "power"),  # Support for "3^2" format
        
        # Root: "Y root of X", "Yth root of X"
        (re.compile(r"^\s*(-?[\d\.]+)\s*(?:st|nd|rd|th)?\s+root\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "root"),
        
        # Square/cube: "square of X", "cube of X"
        (re.compile(r"^\s*square\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "square"),
        (re.compile(r"^\s*cube\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "cube"),
        
        # Square/cube root: "square root of X", "cube root of X"
        (re.compile(r"^\s*square\s+root\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "sqrt"),
        (re.compile(r"^\s*cube\s+root\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "cbrt"),
        
        # Percent: "X percent of Y"
        (re.compile(r"^\s*(-?[\d\.]+)\s+percent\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "percent"),
        
        # Factorial: "factorial of X"
        (re.compile(r"^\s*factorial\s+of\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "factorial"),
        
        # Division expressions: "X divided by Y"
        (re.compile(r"^\s*(-?[\d\.]+)\s+divided\s+by\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "divide"),
        
        # Multiplication expressions: "X times Y", "X multiplied by Y"
        (re.compile(r"^\s*(-?[\d\.]+)\s+times\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "multiply"),
        (re.compile(r"^\s*(-?[\d\.]+)\s+multiplied\s+by\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "multiply"),
        
        # Addition expressions: "X plus Y", "sum of X and Y" 
        (re.compile(r"^\s*(-?[\d\.]+)\s+plus\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "add"),
        (re.compile(r"^\s*sum\s+of\s+(-?[\d\.]+)\s+and\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "add"),
        
        # Subtraction expressions: "X minus Y", "difference of X and Y"
        (re.compile(r"^\s*(-?[\d\.]+)\s+minus\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "subtract"),
        (re.compile(r"^\s*difference\s+(?:between|of)\s+(-?[\d\.]+)\s+and\s+(-?[\d\.]+)\s*$", re.IGNORECASE), "subtract"),
    ]

    def parse(self, query: str) -> Tuple[Optional[Any], Optional[str]]:
        # Normalize query: replace multiple spaces with single space and trim
        normalized_query = re.sub(r'\s+', ' ', query).strip()
        
        # First check for "X% of Y" or "X % of Y" patterns
        percent_match = self.percent_of_pattern.match(normalized_query) or self.alt_percent_pattern.match(normalized_query)
        if percent_match:
            try:
                percent_val = float(percent_match.group(1))
                base_val = float(percent_match.group(2))
                # Calculate X% of Y as (X/100) * Y
                result = (percent_val / 100) * base_val
                final_result = int(result) if result == int(result) else round(result, 10)
                return final_result, None
            except ValueError:
                return None, f"Invalid number in percentage calculation."
            except Exception as e:
                return None, f"An unexpected error in percentage calculation: {e}"
        
        # Next try the traditional pattern with percentage support
        trad_match = self.traditional_math_pattern.match(normalized_query)
        if trad_match:
            num1_str = trad_match.group(1)
            num1_percent = trad_match.group(2) == '%'
            op = trad_match.group(3)
            num2_str = trad_match.group(4)
            num2_percent = trad_match.group(5) == '%'
            
            try:
                num1 = float(num1_str)
                num2 = float(num2_str)
                
                # Convert percentages to decimals if needed
                if num1_percent:
                    num1 = num1 / 100
                if num2_percent:
                    num2 = num2 / 100
                
                return self._calculate(num1, op, num2)
            except ValueError:
                return None, f"Invalid number ('{num1_str}' or '{num2_str}')."
            except OverflowError:
                return None, "Error: Calculation resulted in overflow."
            except Exception as e:
                return None, f"An unexpected math calculation error: {e}"
        
        # If traditional pattern didn't match, try humanized patterns
        for pattern, operation in self.humanized_patterns:
            match = pattern.match(normalized_query)
            if match:
                try:
                    # Handle operations with different argument counts
                    if operation in ["square", "cube", "sqrt", "cbrt", "factorial"]:
                        # Single-argument operations
                        num_str = match.group(1)
                        num = float(num_str)
                        
                        if operation == "square":
                            result = num ** 2
                        elif operation == "cube":
                            result = num ** 3
                        elif operation == "sqrt":
                            if num < 0:
                                return None, "Error: Cannot take square root of negative number."
                            result = math.sqrt(num)
                        elif operation == "cbrt":
                            result = num ** (1/3)
                        elif operation == "factorial":
                            if num < 0 or num != int(num):
                                return None, "Error: Factorial is only defined for non-negative integers."
                            result = math.factorial(int(num))
                            
                        # Format result
                        final_result = int(result) if result == int(result) else round(result, 10)
                        return final_result, None
                    else:
                        # Two-argument operations
                        num1_str, num2_str = match.group(1), match.group(2)
                        num1, num2 = float(num1_str), float(num2_str)
                        
                        if operation == "power":
                            op = "^"
                        elif operation == "root":
                            # For "Y root of X", we calculate X^(1/Y)
                            # Note we swap num1 and num2 here because the pattern captures "Y root of X"
                            return self._calculate(num2, "^", 1/num1)
                        elif operation == "percent":
                            # X percent of Y = X * Y / 100
                            return self._calculate(num1 * num2, "/", 100)
                        elif operation == "divide":
                            op = "/"
                        elif operation == "multiply":
                            op = "*"
                        elif operation == "add":
                            op = "+"
                        elif operation == "subtract":
                            op = "-"
                        else:
                            return None, f"Unimplemented operation: {operation}"
                            
                        return self._calculate(num1, op, num2)
                        
                except ValueError:
                    groups = match.groups()
                    return None, f"Invalid number in expression: {groups}"
                except OverflowError:
                    return None, "Error: Calculation resulted in overflow."
                except Exception as e:
                    return None, f"An unexpected math calculation error: {e}"
        
        # If we get here, none of the patterns matched
        return None, None

    def _calculate(self, num1: float, op: str, num2: float) -> Tuple[Optional[Any], Optional[str]]:
        """Performs the actual calculation based on the operator."""
        result = None
        
        try:
            if op == "+":
                result = num1 + num2
            elif op == "-":
                result = num1 - num2
            elif op == "*":
                result = num1 * num2
            elif op == "/":
                if num2 == 0:
                    return None, "Error: Division by zero."
                result = num1 / num2
            elif op == "%":
                if num2 == 0:
                    return None, "Error: Modulo by zero."
                result = num1 % num2
            elif op == "^":
                result = num1 ** num2
                
            if result is not None:
                # Store numbers as numbers (int or float) for potential reuse
                final_result = int(result) if result == int(result) else round(result, 10)
                return final_result, None
            else:
                return None, "Unknown operator."  # Should not happen
                
        except OverflowError:
            return None, "Error: Calculation resulted in overflow."
        except Exception as e:
            return None, f"An unexpected math calculation error: {e}"


# --- Flask App Setup ---

app = flask.Flask(__name__)
app.config["DEBUG"] = DEBUG_MODE

# --- Calculator Registration ---
# The calculators are ordered by specificity - most specific first, most general last
# This ensures that more specific patterns take precedence over more general ones
REGISTERED_CALCULATORS = [
    TimeCalculator(),  # Time calculations like "9am - 5pm" or "9:30 - 10:45 in minutes"
    UnitCalculator(),  # Unit conversions like "5 km to miles" 
    MathCalculator(),  # Math operations including natural language like "3 power of 2", "square root of 9", "2% of 100"
]

# --- Assignment Pattern ---
# Regex to capture 'var_name = expression_body'
ASSIGNMENT_PATTERN = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$")

# --- API Endpoints ---


@app.route("/calculate", methods=["POST"])
def calculate():
    """Performs a calculation without requiring a session."""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in JSON payload"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # For sessionless calculation, we don't want variable assignments
    if ASSIGNMENT_PATTERN.match(query):
        return jsonify({"error": "Variable assignments require a session. Use /sessions endpoint."}), 400

    # Evaluate the expression using calculators
    calculated_result = None
    evaluation_error = None

    # Iterate through registered calculators
    for calculator in REGISTERED_CALCULATORS:
        try:
            result, error = calculator.parse(query)

            if error:
                # Calculator recognized format but failed internally
                evaluation_error = error
                logger.warning(
                    f"No-session: Calculator {type(calculator).__name__} failed for expression '{query}': {error}"
                )
                break  # Stop trying other calculators if one matched format but failed
            elif result is not None:
                # Calculator successfully parsed and returned a result
                calculated_result = result
                logger.info(
                    f"No-session: Calculator {type(calculator).__name__} succeeded for expression '{query}'. Result: {result}"
                )
                break  # Stop trying once a calculator succeeds
            # else: result is None and error is None, format didn't match. Continue loop.

        except Exception as e:
            # Catch unexpected errors within a calculator's parse method itself
            logger.error(
                f"No-session: Internal error in {type(calculator).__name__} for expr '{query}': {e}",
                exc_info=True,
            )
            evaluation_error = "An internal server error occurred during calculation."
            break  # Stop processing if a calculator has a critical bug

    # Handle results
    if evaluation_error:
        # An error occurred during evaluation (either parse error or internal)
        return jsonify({"error": evaluation_error}), 400
    elif calculated_result is not None:
        # Evaluation succeeded
        return jsonify({"result": calculated_result}), 200
    else:
        # No calculator could handle the expression
        logger.warning(f"No-session: No calculator could parse expression: '{query}'")
        return jsonify({"error": f"Could not understand or parse expression: '{query}'"}), 400


@app.route("/sessions", methods=["POST"])
def create_session():
    """Creates a new calculation session."""
    session_id = create_session_db()
    if session_id:
        return jsonify({"session_id": session_id}), 201  # 201 Created
    else:
        return jsonify({"error": "Failed to create session"}), 500


@app.route("/sessions/<string:session_id>", methods=["GET"])
def get_session_vars(session_id):
    """Retrieves the variables for a given session."""
    variables = load_session(session_id)
    if variables is not None:
        # Filter out internal variables (those starting with underscore)
        public_vars = {k: v for k, v in variables.items() if not k.startswith('_')}
        return jsonify(public_vars), 200
    else:
        return (
            jsonify({"error": f"Session '{session_id}' not found or failed to load"}),
            404,
        )


@app.route("/sessions/<string:session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Deletes a calculation session."""
    if delete_session_db(session_id):
        return "", 204  # No Content
    else:
        # Could be 404 if not found, or 500 if DB error occurred during delete
        return (
            jsonify(
                {"error": f"Failed to delete session '{session_id}' (may not exist)"}
            ),
            404,
        )  # Or 500


@app.route("/sessions/<string:session_id>/calculate", methods=["POST"])
def calculate_in_session(session_id):
    """Performs calculation or assignment within a session."""
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' in JSON payload"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # 1. Load session variables
    variables = load_session(session_id)
    if variables is None:
        return (
            jsonify({"error": f"Session '{session_id}' not found or failed to load"}),
            404,
        )

    # 2. Check for assignment
    assignment_match = ASSIGNMENT_PATTERN.match(query)
    is_assignment = assignment_match is not None
    variable_name = None
    expression_to_evaluate = query

    if is_assignment:
        variable_name = assignment_match.group(1)
        expression_body = assignment_match.group(2).strip()
        if not expression_body:
            return (
                jsonify(
                    {"error": "Assignment requires an expression (e.g., x = 1 + 1)"}
                ),
                400,
            )
            
        # Handle direct number assignment like "x = 10"
        try:
            # Try to interpret as a direct number
            direct_value = float(expression_body)
            # If it's a whole number, convert to int
            if direct_value == int(direct_value):
                direct_value = int(direct_value)
            # Store and return the result
            variables[variable_name] = direct_value
            save_session(session_id, variables)
            return jsonify({"variable_set": variable_name, "result": direct_value}), 200
        except ValueError:
            # Not a direct number, continue with other checks
            pass
            
        # Special handling for unit conversion assignments (e.g., "x = 10km to mi")
        # First check if it's a unit conversion format
        unit_match = UnitCalculator.unit_pattern.match(expression_body)
        if unit_match:
            value_str, from_unit, to_unit = (
                unit_match.group(1),
                unit_match.group(2).strip(),
                unit_match.group(3).strip(),
            )
            try:
                value = float(value_str)
                # Perform the conversion
                calculator = UnitCalculator()
                result, error = calculator.parse(expression_body)
                if error:
                    return jsonify({"error": error}), 400
                if result is not None:
                    # Store both the string result with units and the numeric value
                    variables[variable_name] = result
                    # Extract numeric part for later calculations
                    try:
                        value_str = str(result).split(' ', 1)[0]
                        numeric_value = float(value_str)
                        variables[f"_{variable_name}_value"] = numeric_value
                    except (ValueError, IndexError):
                        pass
                    # Save to the session
                    save_session(session_id, variables)
                    return jsonify({"variable_set": variable_name, "result": result}), 200
            except (ValueError, UndefinedUnitError) as e:
                # Not a valid unit conversion, continue with normal evaluation
                pass
                
        # Substitute variables *only in the expression part*
        expression_to_evaluate = substitute_variables(expression_body, variables)
        logger.info(
            f"Session {session_id}: Assignment detected for '{variable_name}'. Evaluating: '{expression_to_evaluate}'"
        )
    else:
        # Check for operations with unit-based variables (e.g., "x * 3" where x contains units)
        # Patterns like "x * 3", "x / 2", "x + 10", etc.
        unit_op_match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([\+\-\*\/])\s*(-?[\d\.]+)\s*$', query)
        if unit_op_match:
            var_name = unit_op_match.group(1)
            operator = unit_op_match.group(2)
            try:
                number = float(unit_op_match.group(3))
                
                # Check if variable exists and if we have a hidden numeric value for it
                if var_name in variables:
                    hidden_value_key = f"_{var_name}_value"
                    if hidden_value_key in variables:
                        numeric_val = variables[hidden_value_key]
                        
                        # Perform the operation
                        result_val = None
                        if operator == '+':
                            result_val = numeric_val + number
                        elif operator == '-':
                            result_val = numeric_val - number
                        elif operator == '*':
                            result_val = numeric_val * number
                        elif operator == '/':
                            if number == 0:
                                return jsonify({"error": "Division by zero"}), 400
                            result_val = numeric_val / number
                        
                        # If the original variable had units, preserve them
                        if isinstance(variables[var_name], str) and ' ' in variables[var_name]:
                            _, unit = variables[var_name].split(' ', 1)
                            return jsonify({"result": f"{result_val} {unit}"}), 200
                        else:
                            return jsonify({"result": result_val}), 200
            except ValueError:
                # If any conversion fails, continue with normal processing
                pass
        
        # Regular expression with variable substitution
        expression_to_evaluate = substitute_variables(query, variables)
        logger.info(
            f"Session {session_id}: Evaluating expression: '{expression_to_evaluate}' (Original: '{query}')"
        )

    # 3. Evaluate the (potentially substituted) expression using calculators
    calculated_result = None
    evaluation_error = None

    # Iterate through registered calculators
    for calculator in REGISTERED_CALCULATORS:
        try:
            result, error = calculator.parse(expression_to_evaluate)

            if error:
                # Calculator recognized format but failed internally
                evaluation_error = error
                logger.warning(
                    f"Session {session_id}: Calculator {type(calculator).__name__} failed for expression '{expression_to_evaluate}': {error}"
                )
                break  # Stop trying other calculators if one matched format but failed
            elif result is not None:
                # Calculator successfully parsed and returned a result
                calculated_result = result
                logger.info(
                    f"Session {session_id}: Calculator {type(calculator).__name__} succeeded for expression '{expression_to_evaluate}'. Result: {result}"
                )
                break  # Stop trying once a calculator succeeds
            # else: result is None and error is None, format didn't match. Continue loop.

        except Exception as e:
            # Catch unexpected errors within a calculator's parse method itself
            logger.error(
                f"Session {session_id}: Internal error in {type(calculator).__name__} for expr '{expression_to_evaluate}': {e}",
                exc_info=True,
            )
            evaluation_error = "An internal server error occurred during calculation."
            break  # Stop processing if a calculator has a critical bug

    # 4. Handle results and update session if assignment
    if evaluation_error:
        # An error occurred during evaluation (either parse error or internal)
        return (
            jsonify({"error": evaluation_error}),
            400,
        )  # Bad request due to calculation error

    elif calculated_result is not None:
        # Evaluation succeeded
        if is_assignment:
            # Store the result, preserving both numeric value and formatted display
            variables[variable_name] = calculated_result
            
            # Also store a numeric version if it's a unit result
            if isinstance(calculated_result, str) and ' ' in calculated_result:
                try:
                    # Try to extract numeric part for calculations
                    value_str = calculated_result.split(' ', 1)[0]
                    numeric_value = float(value_str)
                    variables[f"_{variable_name}_value"] = numeric_value
                except (ValueError, IndexError):
                    pass
                    
            # Save to database if we have a session
            save_session(session_id, variables)
            
            # Return the assigned value
            return (
                jsonify({"variable_set": variable_name, "result": calculated_result}),
                200,
            )
        else:
            # Just return the result of the expression
            return jsonify({"result": calculated_result}), 200
    else:
        # No calculator could handle the substituted expression
        logger.warning(
            f"Session {session_id}: No calculator could parse substituted expression: '{expression_to_evaluate}' (Original: '{query}')"
        )
        return (
            jsonify(
                {
                    "error": f"Could not understand or parse expression: '{expression_to_evaluate}'"
                }
            ),
            400,
        )


# --- CLI Interface ---
def run_cli_mode():
    """Run calculator in interactive CLI mode."""
    import os
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculator DSL CLI')
    parser.add_argument('--session', '-s', type=str, help='Use an existing session ID')
    parser.add_argument('--keep-session', '-k', action='store_true', help='Keep the session after exiting')
    args = parser.parse_args()
    
    # Variables that will be used by the completer
    variables = {}
    
    # Setup readline for better CLI experience with history and line editing
    try:
        # For Unix-like systems
        import readline
        histfile = os.path.join(os.path.expanduser("~"), ".calc_dsl_history")
        try:
            readline.read_history_file(histfile)
            # Set the max history file size
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
            
        # Save history on exit
        import atexit
        atexit.register(readline.write_history_file, histfile)
        
        # Enable tab completion for commands
        def completer(text, state):
            commands = ['help', 'vars', 'exit', 'quit', 'print', 'keep-session']
            # If there's a space, it might be a 'print' command followed by a variable name
            if ' ' in text:
                cmd, partial_var = text.split(' ', 1)
                if cmd.lower() == 'print':
                    # Complete variable names
                    matching_vars = [f"print {v}" for v in variables.keys() if v.startswith(partial_var) and not v.startswith('_')]
                    if state < len(matching_vars):
                        return matching_vars[state]
                    return None
            # Standard command completion
            matches = [cmd for cmd in commands if cmd.startswith(text)]
            if state < len(matches):
                return matches[state]
            return None
            
        readline.parse_and_bind("tab: complete")
        readline.set_completer(completer)
        
        has_readline = True
    except (ImportError, ModuleNotFoundError):
        # For Windows or if readline is not available
        try:
            import pyreadline3
            has_readline = True
        except (ImportError, ModuleNotFoundError):
            has_readline = False
            
    print("Calculator DSL - Press Ctrl+C to exit")
    print("Enter calculations or expressions like: '2 + 2', '9am - 5pm', or '5 km to miles'")
    if not has_readline:
        print("Note: Install 'readline' (Unix) or 'pyreadline3' (Windows) for command history and arrow key support")
    
    # Initialize database
    init_db()
    
    # Create or load session
    keep_session = args.keep_session
    session_id = None
    
    if args.session:
        # Try to load existing session
        variables = load_session(args.session)
        if variables is not None:
            session_id = args.session
            print(f"Loaded existing session: {session_id}")
        else:
            print(f"Session {args.session} not found. Creating a new session.")
            session_id = create_session_db()
            variables = {}
    else:
        # Create a new session
        session_id = create_session_db()
        variables = {}
    
    if session_id and not args.session:
        print(f"Session created: {session_id}")
        print("Session will be deleted on exit. Use 'keep-session' command to keep it.")
    elif not session_id:
        print("Warning: Failed to create session. Variables won't be saved.")
    
    try:
        while True:
            try:
                # Get user input with the prompt
                query = input("calc> ")
                
                if not query.strip():
                    continue
                
                if query.lower() in ('exit', 'quit', 'bye'):
                    print("Thank you for using Calculator DSL!")
                    if session_id and not keep_session:
                        print(f"Deleting temporary session: {session_id}")
                    elif session_id and keep_session:
                        print(f"Keeping session: {session_id}")
                        print(f"You can resume this session later with: calc --session {session_id}")
                    break
                
                if query.lower() == 'keep-session':
                    keep_session = True
                    print(f"Session {session_id} will be kept after exiting.")
                    print(f"You can resume this session later with: calc --session {session_id}")
                    continue
                
                if query.lower() == 'vars':
                    # Display defined variables
                    if not variables:
                        print("No variables defined.")
                    else:
                        print("Current variables:")
                        for name, value in variables.items():
                            # Skip internal/hidden variables (those starting with underscore)
                            if not name.startswith('_'):
                                print(f"  {name} = {value}")
                    continue
                
                # Check for print variable command (e.g., "?x" or "print x")
                print_var_match = re.match(r'^\s*(?:\?|print\s+)([a-zA-Z_][a-zA-Z0-9_]*)\s*$', query, re.IGNORECASE)
                if print_var_match:
                    var_name = print_var_match.group(1)
                    if var_name in variables:
                        print(f"{var_name} = {variables[var_name]}")
                    else:
                        print(f"Variable '{var_name}' is not defined")
                    continue
                
                if query.lower() == 'help':
                    # Show help
                    print("\nUsage examples:")
                    print("  2 + 3           - Basic arithmetic")
                    print("  x = 10          - Assign a variable")
                    print("  ?x              - Print the value of variable x")
                    print("  print x         - Alternative way to print variable x")
                    print("  9am - 5pm       - Calculate time difference")
                    print("  9am - 5pm in minutes - Calculate time with specific output format")
                    print("  5 km to miles   - Convert units")
                    print("  3 power of 2    - Use natural language math")
                    print("  square root of 16 - Use more complex expressions")
                    print("  2% of 100       - Calculate percentages")
                    print("\nCommands:")
                    print("  vars            - Show all variables")
                    print("  keep-session    - Keep the session after exiting (for later use)")
                    print("  help            - Show this help message")
                    print("  exit/quit       - Exit the calculator")
                    print("\nSession Management:")
                    print("  - By default, sessions are temporary and deleted on exit")
                    print("  - Use 'keep-session' to preserve your session")
                    print("  - To load a saved session: calc --session SESSION_ID")
                    continue
                
                # Process calculation
                # Check for assignment
                assignment_match = ASSIGNMENT_PATTERN.match(query)
                is_assignment = assignment_match is not None
                variable_name = None
                expression_to_evaluate = query

                if is_assignment:
                    variable_name = assignment_match.group(1)
                    expression_body = assignment_match.group(2).strip()
                    if not expression_body:
                        print("Error: Assignment requires an expression (e.g., x = 1 + 1)")
                        continue
                    
                    # Handle direct number assignment like "x = 10"
                    try:
                        # Try to interpret as a direct number
                        direct_value = float(expression_body)
                        # If it's a whole number, convert to int
                        if direct_value == int(direct_value):
                            direct_value = int(direct_value)
                        variables[variable_name] = direct_value
                        if session_id:
                            save_session(session_id, variables)
                        print(f"{variable_name} = {direct_value}")
                        continue
                    except ValueError:
                        # Not a direct number, continue with normal evaluation
                        pass
                    
                    # Special handling for unit conversion assignments (e.g., "x = 10km to mi")
                    # First check if it's a unit conversion format
                    unit_match = UnitCalculator.unit_pattern.match(expression_body)
                    if unit_match:
                        value_str, from_unit, to_unit = (
                            unit_match.group(1),
                            unit_match.group(2).strip(),
                            unit_match.group(3).strip(),
                        )
                        try:
                            value = float(value_str)
                            # Perform the conversion
                            calculator = UnitCalculator()
                            result, error = calculator.parse(expression_body)
                            if error:
                                print(f"Error: {error}")
                                continue
                            if result is not None:
                                # Store both the string result with units and the numeric value
                                variables[variable_name] = result
                                # Extract numeric part for later calculations
                                try:
                                    value_str = str(result).split(' ', 1)[0]
                                    numeric_value = float(value_str)
                                    variables[f"_{variable_name}_value"] = numeric_value
                                except (ValueError, IndexError):
                                    pass
                                if session_id:
                                    save_session(session_id, variables)
                                print(f"{variable_name} = {result}")
                                continue
                        except (ValueError, UndefinedUnitError):
                            # Not a unit conversion or error in conversion, continue with normal evaluation
                            pass
                    
                    # Substitute variables in the expression part
                    expression_to_evaluate = substitute_variables(expression_body, variables)
                else:
                    # Check for operations with unit-based variables (e.g., "x * 3" where x contains units)
                    # Patterns like "x * 3", "x / 2", "x + 10", etc.
                    unit_op_match = re.match(r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([\+\-\*\/])\s*(-?[\d\.]+)\s*$', query)
                    if unit_op_match:
                        var_name = unit_op_match.group(1)
                        operator = unit_op_match.group(2)
                        number = float(unit_op_match.group(3))
                        
                        # Check if we have a saved numeric value for this variable
                        hidden_value_key = f"_{var_name}_value"
                        if hidden_value_key in variables:
                            numeric_val = variables[hidden_value_key]
                            
                            # Perform the operation
                            if operator == '+':
                                result_val = numeric_val + number
                            elif operator == '-':
                                result_val = numeric_val - number
                            elif operator == '*':
                                result_val = numeric_val * number
                            elif operator == '/':
                                if number == 0:
                                    print("Error: Division by zero")
                                    continue
                                result_val = numeric_val / number
                            
                            # If the original variable had units, preserve them
                            if var_name in variables and isinstance(variables[var_name], str) and ' ' in variables[var_name]:
                                _, unit = variables[var_name].split(' ', 1)
                                print(f"Result: {result_val} {unit}")
                            else:
                                print(f"Result: {result_val}")
                            continue
                            
                    # Regular expression with variable substitution
                    expression_to_evaluate = substitute_variables(query, variables)

                # Evaluate with calculators
                calculated_result = None
                evaluation_error = None

                # Try each calculator
                for calculator in REGISTERED_CALCULATORS:
                    try:
                        result, error = calculator.parse(expression_to_evaluate)

                        if error:
                            # Calculator recognized format but failed internally
                            evaluation_error = error
                            break
                        elif result is not None:
                            # Calculator successfully parsed and returned a result
                            calculated_result = result
                            break

                    except Exception as e:
                        evaluation_error = f"An unexpected error occurred: {e}"
                        break

                # Handle results
                if evaluation_error:
                    print(f"Error: {evaluation_error}")
                elif calculated_result is not None:
                    if is_assignment:
                        # Store the result, preserving both numeric value and formatted display
                        variables[variable_name] = calculated_result
                        print(f"{variable_name} = {calculated_result}")
                        
                        # Also store a numeric version if it's a unit result
                        if isinstance(calculated_result, str) and ' ' in calculated_result:
                            try:
                                # Try to extract numeric part for calculations
                                value_str = calculated_result.split(' ', 1)[0]
                                numeric_value = float(value_str)
                                variables[f"_{variable_name}_value"] = numeric_value
                            except (ValueError, IndexError):
                                pass
                        
                        # Save to database if we have a session
                        if session_id:
                            save_session(session_id, variables)
                    else:
                        print(f"Result: {calculated_result}")
                else:
                    print(f"Error: Could not understand or parse expression: '{expression_to_evaluate}'")
                    
            except Exception as e:
                print(f"Error processing input: {e}")
                
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")
        if session_id and not keep_session:
            print(f"Deleting temporary session: {session_id}")
        elif session_id and keep_session:
            print(f"Keeping session: {session_id}")
            print(f"You can resume this session later with: calc --session {session_id}")
    
    # Clean up the session if needed
    if session_id and not keep_session:
        try:
            delete_session_db(session_id)
        except Exception as e:
            print(f"Warning: Failed to delete session: {e}")
    
    print("Thank you for using Calculator DSL!")

# --- Entry Points ---
def start_web_server():
    """Entry point for running the web server."""
    print("Initializing database...")
    init_db()  # Ensure database exists and table is created on startup
    
    print("Starting web server on http://0.0.0.0:5200")
    app.run(host="0.0.0.0", port=5200)

def start_cli_mode():
    """Entry point for running the CLI mode."""
    # This directly runs the CLI mode
    init_db()
    run_cli_mode()

def main():
    """Main entry point that decides between web and CLI mode based on arguments."""
    import sys
    
    init_db()  # Ensure database exists and table is created on startup
    
    # Check if command line args exist and we should run in CLI mode
    if len(sys.argv) > 1:
        cli_handled = run_cli_mode()
        if cli_handled:
            return
    
    # Default to web server mode if not handled by CLI
    print("Starting web server on http://0.0.0.0:5200")
    app.run(host="0.0.0.0", port=5200)

if __name__ == "__main__":
    main()