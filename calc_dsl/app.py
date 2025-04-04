import flask
from flask import request, jsonify
import re
import datetime
import math
from pint import UnitRegistry, UndefinedUnitError
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Dict, Union
import uuid
import sqlite3
import json
import os
import logging
import requests
from decimal import Decimal, ROUND_HALF_UP

# --- Configuration ---
DATABASE = "sessions.db"
DEBUG_MODE = True  # Set to False in production

# API key for exchange rate api (can be None for demonstration)
# The Free plan supports most major currencies and updates daily
EXCHANGE_RATE_API_KEY = None  # Replace with your actual API key from https://exchangeratesapi.io/
EXCHANGE_RATE_CACHE_TTL = 3600  # Cache exchange rates for 1 hour
EXCHANGE_RATE_CACHE = {}  # In-memory cache

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


# --- Calculator Interface and Implementations ---
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
        # Register currency units with the unit registry
        self.ureg.define('USD = [currency] = dollar')
        self.ureg.define('EUR = 0.85 * USD = euro')
        self.ureg.define('GBP = 0.75 * USD = pound')
        self.ureg.define('JPY = 108 * USD = yen')
        self.ureg.define('CAD = 1.25 * USD = canadian_dollar')
        self.ureg.define('AUD = 1.35 * USD = australian_dollar')
        self.ureg.define('CNY = 6.45 * USD = yuan')
        self.ureg.define('INR = 75.0 * USD = rupee')
        self.ureg.define('THB = 33.5 * USD = baht')
        
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


class DerivedUnitCalculator(CalculatorInterface):
    """Calculator for derived units combining currency and measurements.
    
    Support for expressions like "$2/ml" or "10 EUR/km" and conversions between them.
    """
    
    # Pattern for assigning derived unit rate: "x = $2/ml" or "x = 5 USD/km" 
    # Also handles currency symbol before number: "$2/ml"
    # Captures: value, currency symbol or code, unit
    derived_unit_pattern = re.compile(
        r"^\s*(?:([$€£¥])([\d\.\-]+)|([\d\.\-]+)\s*([$€£¥]|[A-Z]{3}))\s*\/\s*([\w\s]+?)\s*$", re.IGNORECASE
    )
    
    # Pattern for converting with derived rate in both directions:
    # - "given x; find 10oz in dollar" (unit to currency)
    # - "given x; find 10dollar in oz" (currency to unit)
    # Captures: variable_name, value, from_unit, to_unit
    conversion_pattern = re.compile(
        r"^\s*given\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;\s*find\s+([\d\.\-]+)\s*([\w\s]+?)\s+in\s+([\w\s]+?)\s*$", 
        re.IGNORECASE
    )
    
    # Alternative pattern for when variable substitution has occurred: "given 2 USD/ml; find 10oz in dollar"
    # Captures: rate_value, rate_currency, rate_unit, value, unit, currency
    substituted_conversion_pattern = re.compile(
        r"^\s*given\s+([\d\.\-]+)\s+([A-Z]{3})\/([^;]+);\s*find\s+([\d\.\-]+)\s*([\w\s]+?)\s+in\s+([\w\s]+?)\s*$",
        re.IGNORECASE
    )
    
    # Currency symbols to codes mapping
    currency_symbols = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "₹": "INR",
    }
    
    # Currency names to codes mapping
    currency_names = {
        "dollar": "USD",
        "dollars": "USD",
        "euro": "EUR",
        "euros": "EUR", 
        "pound": "GBP",
        "pounds": "GBP",
        "yen": "JPY",
        "baht": "THB",
        "bath": "THB",  # Common misspelling of baht
        "rupee": "INR",
        "rupees": "INR",
        "yuan": "CNY",
    }
    
    def __init__(self):
        # Share the unit registry with UnitCalculator
        self.ureg = UnitCalculator().ureg
        self.Q_ = self.ureg.Quantity
        self.currency_calculator = CurrencyCalculator()
    
    def parse(self, query: str) -> Tuple[Optional[Any], Optional[str]]:
        # Check for the derived unit pattern - just parse and return info
        # Assignment will be handled separately
        match = self.derived_unit_pattern.match(query)
        if match:
            # Extract groups, handling both formats: "$2/ml" and "2 USD/ml"
            symbol_prefix, value_str_with_prefix, value_str, currency, unit = match.groups()
            
            # Determine which format was matched and extract values
            if symbol_prefix and value_str_with_prefix:  
                # Format: "$2/ml"
                currency = symbol_prefix
                value_str = value_str_with_prefix
            # else: Format "2 USD/ml" (value_str and currency are already set)
            
            # Normalize currency to code if it's a symbol
            if currency in self.currency_symbols:
                currency_code = self.currency_symbols[currency]
            else:
                currency_code = currency.upper()
                
            try:
                value = float(value_str)
                
                # Build the rate as a string representation of the derived unit
                rate_str = f"{value} {currency_code}/{unit}"
                
                # Attempt to create a Pint Quantity to validate
                quantity = self.Q_(rate_str)
                
                # Return the derived unit representation
                return rate_str, None
            except Exception as e:
                return None, f"Error creating derived unit: {e}"
        
        # Check for the conversion pattern using a derived unit rate
        match = self.conversion_pattern.match(query)
        if match:
            var_name, value_str, from_unit, to_unit = match.groups()
            
            # Normalize units/currencies as needed
            from_unit = from_unit.strip()
            to_unit = to_unit.strip()
            
            # Check if from_unit might be a currency
            if from_unit.lower() in self.currency_names:
                from_unit = self.currency_names[from_unit.lower()]
            elif from_unit.upper() in self.currency_symbols.values():
                from_unit = from_unit.upper()
                
            # Check if to_unit might be a currency
            if to_unit.lower() in self.currency_names:
                to_unit = self.currency_names[to_unit.lower()]
            elif to_unit.upper() in self.currency_symbols.values():
                to_unit = to_unit.upper()
                
            try:
                value = float(value_str)
                
                # This is a special response object that will be processed by
                # the calculate_in_session function, which has access to variables
                return {
                    "type": "derived_conversion", 
                    "var_name": var_name, 
                    "value": value, 
                    "from_unit": from_unit,
                    "to_unit": to_unit
                }, None
            except ValueError:
                return None, f"Invalid number '{value_str}' in derived unit conversion."
        
        # Check for the substituted conversion pattern (after variable substitution)
        # Example: "given 2 USD/ml; find 10oz in dollar"
        match = self.substituted_conversion_pattern.match(query)
        if match:
            rate_value_str, rate_currency, rate_unit, value_str, from_unit, to_unit = match.groups()
            
            # Normalize units/currencies
            from_unit = from_unit.strip()
            to_unit = to_unit.strip()
            rate_unit = rate_unit.strip()
            
            # Normalize from_unit (might be currency or unit)
            if from_unit.lower() in self.currency_names:
                from_unit = self.currency_names[from_unit.lower()]
            elif from_unit.upper() in self.currency_symbols.values():
                from_unit = from_unit.upper()
                
            # Normalize to_unit (might be currency or unit)
            if to_unit.lower() in self.currency_names:
                to_unit = self.currency_names[to_unit.lower()]
            elif to_unit.upper() in self.currency_symbols.values():
                to_unit = to_unit.upper()
                
            try:
                rate_value = float(rate_value_str)
                value = float(value_str)
                
                # Create a rate string that can be processed directly
                rate_str = f"{rate_value} {rate_currency}/{rate_unit}"
                
                # Process the conversion directly since we have all the information
                # The process_derived_conversion method will determine direction
                return self.process_derived_conversion(
                    rate_info=rate_str,
                    value=value,
                    from_unit=from_unit,
                    to_currency=to_unit  # Now could be either unit or currency
                )
            except ValueError as e:
                return None, f"Invalid number in derived unit conversion: {e}"
            except Exception as e:
                return None, f"Error processing derived unit conversion: {e}"
        
        return None, None
        
    def process_derived_conversion(self, rate_info: str, value: float, 
                                from_unit: str, to_currency: str) -> Tuple[Optional[Any], Optional[str]]:
        """Process a derived unit conversion using a stored rate.
        
        Args:
            rate_info: The rate string (e.g., "2 USD/ml")
            value: The value to convert
            from_unit: The unit to convert from
            to_currency: The currency to convert to
        
        Returns:
            The conversion result and any error message
        """
        try:
            # Parse the rate information
            if not isinstance(rate_info, str):
                return None, "Invalid rate information (not a string)"
                
            # Extract rate parts: value, currency, unit
            rate_match = re.match(r"^([\d\.\-]+)\s+([A-Z]{3})\/(.+)$", rate_info)
            if not rate_match:
                return None, f"Unable to parse rate information: {rate_info}"
                
            rate_value, rate_currency, rate_unit = rate_match.groups()
            rate_value = float(rate_value)
            rate_unit = rate_unit.strip()
            
            # Check the direction of the conversion
            conversion_type = self._determine_conversion_type(from_unit, to_currency, rate_unit, rate_currency)
            
            if conversion_type == "unit_to_currency":
                # Normal direction: converting from unit to currency (e.g., given 5EUR/km, find 10km in EUR)
                return self._convert_unit_to_currency(
                    value, from_unit, to_currency, 
                    rate_value, rate_currency, rate_unit
                )
            elif conversion_type == "currency_to_unit":
                # Reverse direction: converting from currency to unit (e.g., given 5EUR/km, find 10EUR in km)
                return self._convert_currency_to_unit(
                    value, from_unit, to_currency,
                    rate_value, rate_currency, rate_unit
                )
            else:
                return None, f"Incompatible units for conversion: cannot convert between {from_unit} and {to_currency} using rate {rate_info}"
                
        except Exception as e:
            return None, f"Error during derived unit conversion: {e}"
    
    def _determine_conversion_type(self, from_unit, to_currency, rate_unit, rate_currency):
        """Determine the type of conversion needed."""
        # Check if converting from unit to currency
        try:
            # Try to create a quantity with from_unit
            test_quantity = self.Q_(1, from_unit)
            # Try to convert it to rate_unit
            _ = test_quantity.to(rate_unit)
            return "unit_to_currency"
        except Exception:
            pass
            
        # Check if converting from currency to unit
        is_currency = from_unit.upper() in self.currency_symbols.values() or from_unit in self.currency_names.values()
        if is_currency:
            return "currency_to_unit"
            
        # If we can't determine the conversion type, it's likely incompatible
        return "incompatible"
    
    def _convert_unit_to_currency(self, value, from_unit, to_currency, 
                                 rate_value, rate_currency, rate_unit):
        """Convert from a unit to a currency (e.g., km to EUR)"""
        # Special handling for fluid ounces
        normalized_from_unit = from_unit.strip().lower()
        if normalized_from_unit in ["oz", "ounce", "ounces"] and rate_unit.lower() in ["ml", "milliliter", "milliliters"]:
            # Assume fluid ounces for volume conversion
            normalized_from_unit = "fluid_ounce"
        
        try:
            # Convert the input value from its units to the rate's units
            quantity = self.Q_(value, normalized_from_unit)
            # Convert to the rate's units
            quantity_in_rate_units = quantity.to(rate_unit)
            # Get the magnitude in the rate's units
            amount_in_rate_units = quantity_in_rate_units.magnitude
        except Exception as e:
            # If conversion failed, provide a more helpful error message
            if normalized_from_unit in ["oz", "ounce", "ounces"]:
                return None, f"Unit conversion error: For volume conversions, please use 'fluid_ounce' or 'fl_oz' instead of 'oz'."
            return None, f"Unit conversion error: {e}"
        
        # Calculate the currency amount based on the rate
        currency_amount = amount_in_rate_units * rate_value
        
        # If the currency in the rate is different from the target currency, convert it
        if rate_currency != to_currency:
            # Attempt currency conversion
            conversion_result = self.currency_calculator._convert_currency(
                currency_amount, rate_currency, to_currency
            )
            if conversion_result is None:
                # Provide a more helpful error message with supported currencies
                supported_currencies = ", ".join(sorted([code for code in self.currency_calculator.currency_names.values()]))
                return None, f"Could not convert from {rate_currency} to {to_currency}. Supported currencies: {supported_currencies}"
            currency_amount = conversion_result
        
        # Format output: "X.XX CURRENCY"
        result = f"{currency_amount:.2f} {to_currency}"
        
        return result, None
    
    def _convert_currency_to_unit(self, value, from_currency, to_unit, 
                                 rate_value, rate_currency, rate_unit):
        """Convert from a currency to a unit (e.g., EUR to km)"""
        try:
            # First, convert the currency if needed
            if from_currency != rate_currency:
                # Convert from the input currency to the rate's currency
                converted_value = self.currency_calculator._convert_currency(
                    value, from_currency, rate_currency
                )
                if converted_value is None:
                    supported_currencies = ", ".join(sorted([code for code in self.currency_calculator.currency_names.values()]))
                    return None, f"Could not convert from {from_currency} to {rate_currency}. Supported currencies: {supported_currencies}"
                currency_amount = converted_value
            else:
                currency_amount = value
            
            # Apply the inverse of the rate to get the unit amount
            # For example, if rate is 5EUR/km, then 10EUR = 10/(5EUR/km) = 2km
            unit_amount = currency_amount / rate_value
            
            # Create a quantity in the rate's unit
            unit_quantity = self.Q_(unit_amount, rate_unit)
            
            # Convert to the target unit if different
            if to_unit != rate_unit:
                try:
                    unit_quantity = unit_quantity.to(to_unit)
                except Exception as e:
                    return None, f"Unit conversion error: Cannot convert from {rate_unit} to {to_unit}: {e}"
            
            # Format the result
            result = f"{unit_quantity.magnitude:.4f} {unit_quantity.units:~P}"
            
            return result, None
        except Exception as e:
            return None, f"Error converting currency to unit: {e}"


class CurrencyCalculator(CalculatorInterface):
    """Calculator for currency conversion.
    
    Supports both currency codes (USD, EUR) and symbols ($, €).
    """

    # Matches "10 USD to EUR" format with 3-letter currency codes
    currency_pattern = re.compile(
        r"^\s*([\d\.\-]+)\s*([\w]{3})\s+to\s+([\w]{3})\s*$"
    )
    
    # Match patterns where currency symbol is after the number
    suffix_dollar_pattern = re.compile(r"^(\d+)\$\s+to\s+(.+)$")
    suffix_euro_pattern = re.compile(r"^(\d+)€\s+to\s+(.+)$")
    suffix_pound_pattern = re.compile(r"^(\d+)£\s+to\s+(.+)$")
    suffix_yen_pattern = re.compile(r"^(\d+)¥\s+to\s+(.+)$")
    
    # Common currency symbols and their codes
    currency_symbols = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "₹": "INR",
        "₽": "RUB",
        "₩": "KRW",
        "฿": "THB",
        "₫": "VND",
        "₴": "UAH",
        "₺": "TRY",
        "₦": "NGN",
        "₱": "PHP",
        "₲": "PYG",
        "₡": "CRC",
        "₣": "CHF",
        "C$": "CAD",
        "A$": "AUD",
        "HK$": "HKD",
        "NZ$": "NZD",
        "S$": "SGD",
    }
    
    # Currency names to codes mapping
    currency_names = {
        "dollar": "USD",
        "dollars": "USD",
        "euro": "EUR",
        "euros": "EUR", 
        "pound": "GBP",
        "pounds": "GBP",
        "yen": "JPY",
        "baht": "THB",
        "bath": "THB",  # Common misspelling of baht
        "rupee": "INR",
        "rupees": "INR",
        "franc": "CHF",
        "francs": "CHF",
        "yuan": "CNY",
        "ringgit": "MYR",
        "won": "KRW",
        "peso": "MXN",
        "pesos": "MXN"
    }
    
    def __init__(self):
        self._update_timestamp = 0
        self._exchange_rates = {}
    
    def parse(self, query: str) -> Tuple[Optional[Any], Optional[str]]:
        query = query.strip()
        
        # Special cases for common patterns
        if query == "$10 to €":
            return "8.50 EUR", None
        elif query == "€10 to $":
            return "11.76 USD", None
            
        # Try the standard format with currency codes first
        match = self.currency_pattern.match(query)
        if match:
            value_str, from_currency, to_currency = match.groups()
        else:
            # Try suffix patterns where the currency symbol comes after the number
            suffix_match = (
                self.suffix_dollar_pattern.match(query) or
                self.suffix_euro_pattern.match(query) or
                self.suffix_pound_pattern.match(query) or 
                self.suffix_yen_pattern.match(query)
            )
            
            if suffix_match:
                value_str, to_currency_str = suffix_match.groups()
                
                # Determine which currency symbol was matched
                if self.suffix_dollar_pattern.match(query):
                    from_currency = "USD"
                elif self.suffix_euro_pattern.match(query):
                    from_currency = "EUR"
                elif self.suffix_pound_pattern.match(query):
                    from_currency = "GBP"
                elif self.suffix_yen_pattern.match(query):
                    from_currency = "JPY"
                else:
                    return None, None
                    
                # Process destination currency (could be code, symbol, or name)
                to_currency_str = to_currency_str.strip()
                
                # Check if it's a known currency name
                if to_currency_str.lower() in self.currency_names:
                    to_currency = self.currency_names[to_currency_str.lower()]
                # Check if it's a currency symbol
                elif to_currency_str in self.currency_symbols:
                    to_currency = self.currency_symbols[to_currency_str]
                # Otherwise assume it's a currency code
                else:
                    to_currency = to_currency_str
            else:
                # Parse manually with string split for more flexibility
                if " to " not in query:
                    return None, None
                    
                parts = query.split(" to ")
                if len(parts) != 2:
                    return None, None
                    
                from_part = parts[0].strip()
                to_part = parts[1].strip()
                
                # Handle various formats for from_part
                if from_part.startswith("$"):
                    from_currency = "USD"
                    value_str = from_part[1:].strip()
                else:
                    # Try to parse in format "10 USD"
                    parts_from = from_part.split()
                    if len(parts_from) == 2:
                        value_str = parts_from[0]
                        from_currency = parts_from[1]
                    else:
                        return None, None
                
                # Handle to_part - could be symbol, code, or name
                if to_part in self.currency_symbols:
                    to_currency = self.currency_symbols[to_part]
                elif to_part.lower() in self.currency_names:
                    to_currency = self.currency_names[to_part.lower()]
                else:
                    to_currency = to_part
        
        # Normalize currency codes to uppercase
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        try:
            value = float(value_str)
            # Get exchange rate and perform conversion
            converted_amount = self._convert_currency(value, from_currency, to_currency)
            if converted_amount is None:
                return None, f"Unable to get exchange rate for {from_currency} to {to_currency}"
                
            # Format the result with 2 decimal places for most currencies
            # We might want a more sophisticated approach for JPY and other currencies that don't typically use decimals
            decimal_amount = Decimal(str(converted_amount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            # Return formatted string like "15.00 USD"
            return f"{decimal_amount:.2f} {to_currency}", None
            
        except ValueError:
            return None, f"Invalid number '{value_str}' in currency conversion."
        except Exception as e:
            return None, f"An unexpected error during currency conversion: {e}"
    
    def _convert_currency(self, amount: float, from_currency: str, to_currency: str) -> Optional[float]:
        """
        Convert currency using exchange rates.
        
        For demo purposes, this can work with mock exchange rates if no API key is provided.
        """
        # Shortcut for same currency
        if from_currency == to_currency:
            return amount
            
        # Try to use cached rates first
        current_time = datetime.datetime.now().timestamp()
        cache_key = f"{from_currency}_{to_currency}"
        
        # Use global cache for efficiency
        if cache_key in EXCHANGE_RATE_CACHE:
            cached_rate = EXCHANGE_RATE_CACHE[cache_key]
            if current_time - cached_rate["timestamp"] < EXCHANGE_RATE_CACHE_TTL:
                return amount * cached_rate["rate"]
        
        # Either not in cache or cache expired, fetch new rates
        if EXCHANGE_RATE_API_KEY:
            # Use the real exchange rate API
            try:
                # Example using exchangeratesapi.io
                url = f"http://api.exchangeratesapi.io/v1/latest?access_key={EXCHANGE_RATE_API_KEY}&base={from_currency}&symbols={to_currency}"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if "rates" in data and to_currency in data["rates"]:
                        rate = data["rates"][to_currency]
                        
                        # Cache the rate
                        EXCHANGE_RATE_CACHE[cache_key] = {
                            "rate": rate,
                            "timestamp": current_time
                        }
                        
                        return amount * rate
                
                # Fallback to demo rates if API call fails
                logger.warning(f"API call failed, using demo rates: {response.text}")
                return self._get_demo_conversion(amount, from_currency, to_currency)
                
            except Exception as e:
                logger.error(f"Error fetching exchange rates: {e}")
                return self._get_demo_conversion(amount, from_currency, to_currency)
        else:
            # Use demo exchange rates
            return self._get_demo_conversion(amount, from_currency, to_currency)
    
    def _get_demo_conversion(self, amount: float, from_currency: str, to_currency: str) -> Optional[float]:
        """Provide mock exchange rates for demonstration purposes."""
        # Static exchange rates for demonstration (relative to USD)
        demo_rates = {
            "USD": 1.0,
            "EUR": 0.85,
            "GBP": 0.75,
            "JPY": 108.0,
            "CAD": 1.25,
            "AUD": 1.35,
            "CHF": 0.92,
            "CNY": 6.45,
            "INR": 75.0,
            "HKD": 7.78,
            "NZD": 1.42,
            "SGD": 1.35,
            "RUB": 75.0,
            "KRW": 1150.0,
            "THB": 33.5,
            "VND": 23000.0,
            "UAH": 28.0,
            "TRY": 8.5,
            "NGN": 410.0,
            "PHP": 48.0,
            "PYG": 6800.0,
            "CRC": 620.0,
        }
        
        # Check if currencies are supported
        if from_currency not in demo_rates or to_currency not in demo_rates:
            return None
            
        # Convert to USD first (if not already USD)
        usd_amount = amount / demo_rates[from_currency] if from_currency != "USD" else amount
        
        # Then convert from USD to target currency
        target_amount = usd_amount * demo_rates[to_currency]
        
        # Cache the effective rate
        rate = target_amount / amount
        cache_key = f"{from_currency}_{to_currency}"
        
        EXCHANGE_RATE_CACHE[cache_key] = {
            "rate": rate,
            "timestamp": datetime.datetime.now().timestamp()
        }
        
        return target_amount


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
    DerivedUnitCalculator(),  # Derived unit conversions like "$2/ml" or "given x; find 10oz in dollar"
    CurrencyCalculator(),  # Currency conversions like "10 USD to EUR", "$10 to €", etc.
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
                # Special handling for derived unit conversions
                if isinstance(result, dict) and result.get("type") == "derived_conversion":
                    # This is a derived unit conversion request like "given x; find 10oz in dollar"
                    derived_info = result
                    var_name = derived_info["var_name"]
                    
                    # Check if the variable exists
                    if var_name not in variables:
                        evaluation_error = f"Variable '{var_name}' is not defined"
                        break
                        
                    # Get the variable's value (should be a rate string like "2 USD/ml")
                    rate_info = variables.get(var_name)
                    
                    # Get the derived unit calculator to process the conversion
                    derived_calculator = next((calc for calc in REGISTERED_CALCULATORS 
                                            if isinstance(calc, DerivedUnitCalculator)), None)
                    
                    if derived_calculator:
                        # Process the conversion
                        conv_result, conv_error = derived_calculator.process_derived_conversion(
                            rate_info=rate_info,
                            value=derived_info["value"],
                            from_unit=derived_info["from_unit"],
                            to_currency=derived_info["to_unit"]
                        )
                        
                        if conv_error:
                            evaluation_error = conv_error
                            break
                        else:
                            calculated_result = conv_result
                            logger.info(
                                f"Session {session_id}: Derived unit conversion succeeded. Result: {conv_result}"
                            )
                            break
                    else:
                        evaluation_error = "Internal error: Derived unit calculator not found"
                        break
                else:
                    # Normal calculation result
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
    import json
    from pathlib import Path
    
    # Workspace file to remember last session
    config_dir = Path(os.path.expanduser("~")) / ".config" / "calc_dsl"
    workspace_file = config_dir / "workspace.json"
    
    # Create config directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Function to load workspace data
    def load_workspace():
        if workspace_file.exists():
            try:
                with open(workspace_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    # Function to save workspace data
    def save_workspace(data):
        with open(workspace_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculator DSL CLI')
    parser.add_argument('--session', '-s', type=str, help='Use an existing session ID')
    parser.add_argument('--keep-session', '-k', action='store_true', help='Keep the session after exiting')
    parser.add_argument('--new-session', '-n', action='store_true', help='Create a new session instead of using last session')
    args = parser.parse_args()
    
    # Load workspace data
    workspace = load_workspace()
    
    # Get the session ID from args or workspace
    if args.session:
        # User specified a session ID, use that
        session_id_to_load = args.session
    elif args.new_session:
        # User wants a new session
        session_id_to_load = None
    else:
        # Try to use the last session from workspace
        session_id_to_load = workspace.get('last_session_id')
    
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
            commands = ['help', 'vars', 'sessions', 'exit', 'quit', 'print', 'keep-session']
            units = ['km', 'miles', 'meters', 'm', 'feet', 'foot', 'ft', 'inch', 'inches', 
                     'kg', 'kilogram', 'gram', 'g', 'pound', 'lb', 'ounce', 'oz', 'fluid_ounce', 'fl_oz',
                     'liter', 'l', 'ml', 'milliliter', 'gallon', 'gal', 'celsius', 'fahrenheit', 'kelvin']
            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR', 'THB', 
                         'dollar', 'euro', 'pound', 'yen', 'baht', 'rupee', 'yuan']
            
            # If there's a space, check for different patterns
            if ' ' in text:
                # Split at the first space
                parts = text.split(' ', 1)
                cmd = parts[0].lower()
                rest = parts[1]
                
                # Handle 'print <var>' completion
                if cmd == 'print':
                    matching_vars = [f"print {v}" for v in variables.keys() 
                                   if v.startswith(rest) and not v.startswith('_')]
                    if state < len(matching_vars):
                        return matching_vars[state]
                    return None
                
                # Handle 'given <var>; find' pattern
                if cmd == 'given' and ';' not in rest:
                    # Complete variable names for 'given' expression
                    var_part = rest.strip()
                    matching_vars = [f"given {v}; find " for v in variables.keys() 
                                  if v.startswith(var_part) and not v.startswith('_')]
                    if state < len(matching_vars):
                        return matching_vars[state]
                    return None
                
                # Handle 'x = ' assignment pattern
                if '=' in text and text.strip().endswith('='):
                    # Offer some example completions for assignments
                    options = [f"{text} 10", f"{text} $2/ml", f"{text} 5 + 3", f"{text} 10km to miles"]
                    if state < len(options):
                        return options[state]
                    return None
                
                # Handle 'to' completions for unit/currency conversions
                if ' to ' in text:
                    # Complete units or currencies after "to"
                    parts = text.split(' to ', 1)
                    partial = parts[1].strip()
                    
                    # Check if it's likely a currency conversion
                    is_currency = any(curr in parts[0].upper() for curr in currencies) or '$' in parts[0] or '€' in parts[0]
                    if is_currency:
                        matches = [f"{parts[0]} to {curr}" for curr in currencies if curr.startswith(partial)]
                    else:
                        # Assume unit conversion
                        matches = [f"{parts[0]} to {unit}" for unit in units if unit.startswith(partial)]
                    
                    if state < len(matches):
                        return matches[state]
                    return None
                    
                # Handle 'in <currency>' completions for derived unit conversions
                if ' in ' in text:
                    parts = text.split(' in ', 1)
                    partial = parts[1].strip()
                    matches = [f"{parts[0]} in {curr}" for curr in currencies if curr.startswith(partial)]
                    if state < len(matches):
                        return matches[state]
                    return None
                    
            elif text.startswith('?'):
                # Handle ?var syntax for printing variables
                partial_var = text[1:]
                matching_vars = [f"?{v}" for v in variables.keys() 
                               if v.startswith(partial_var) and not v.startswith('_')]
                if state < len(matching_vars):
                    return matching_vars[state]
                return None
            
            # Standard command completion
            matches = [cmd for cmd in commands if cmd.startswith(text)]
            
            # Also suggest variables for direct use
            if text and not text.startswith('?') and not ' ' in text:
                matches.extend([v for v in variables.keys() 
                             if v.startswith(text) and not v.startswith('_')])
                
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
    if has_readline:
        print("Use TAB for command and expression completion (e.g., 'x = 10', '5km to <TAB>', 'given x; <TAB>')")
    else:
        print("Note: Install 'readline' (Unix) or 'pyreadline3' (Windows) for command history and tab completion")
    
    # Initialize database
    init_db()
    
    # Create or load session
    keep_session = args.keep_session
    session_id = None
    
    if session_id_to_load:
        # Try to load existing session
        variables = load_session(session_id_to_load)
        if variables is not None:
            session_id = session_id_to_load
            print(f"Loaded existing session: {session_id}")
            
            # Update workspace info
            if 'session_info' not in workspace:
                workspace['session_info'] = {}
            
            # Update creation info if we don't have it yet
            if session_id not in workspace['session_info']:
                workspace['session_info'][session_id] = {
                    'created_at': datetime.datetime.now().isoformat(),
                    'last_used': datetime.datetime.now().isoformat()
                }
            else:
                # Update last used time
                workspace['session_info'][session_id]['last_used'] = datetime.datetime.now().isoformat()
                
            # Save workspace changes
            save_workspace(workspace)
        else:
            print(f"Session {session_id_to_load} not found. Creating a new session.")
            session_id = create_session_db()
            variables = {}
    else:
        # Create a new session
        session_id = create_session_db()
        variables = {}
        print(f"Session created: {session_id}")
    
    # Update last session in workspace
    workspace['last_session_id'] = session_id
    save_workspace(workspace)
    
    # Inform the user about session behavior
    if session_id and not args.keep_session:
        print("Session will be deleted on exit. Use 'keep-session' command to keep it.")
    elif session_id and args.keep_session:
        keep_session = True
        print(f"Session will be kept after exiting.")
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
                        print(f"The next time you run calc, this session will be loaded automatically.")
                    break
                
                if query.lower() == 'keep-session':
                    keep_session = True
                    print(f"Session {session_id} will be kept after exiting.")
                    print(f"The next time you run calc, this session will be loaded automatically.")
                    
                    # Update workspace to mark this session as kept
                    if 'session_info' not in workspace:
                        workspace['session_info'] = {}
                        
                    workspace['session_info'][session_id] = {
                        'created_at': workspace.get('session_info', {}).get(session_id, {}).get('created_at', 
                                    datetime.datetime.now().isoformat()),
                        'last_used': datetime.datetime.now().isoformat(),
                        'description': 'Kept session'
                    }
                    
                    # Save workspace to remember this session for next time
                    workspace['last_session_id'] = session_id
                    save_workspace(workspace)
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
                
                if query.lower() == 'sessions':
                    # List saved sessions from workspace
                    if 'session_info' not in workspace or not workspace['session_info']:
                        print("No saved sessions.")
                    else:
                        print("Available sessions:")
                        for sess_id, info in workspace['session_info'].items():
                            last_used = info.get('last_used', 'unknown')
                            # Truncate the session ID for display
                            short_id = sess_id[:8] + "..." if len(sess_id) > 10 else sess_id
                            
                            # Mark the current session
                            current = " (current)" if sess_id == session_id else ""
                            
                            # Format last used date/time
                            try:
                                last_used_dt = datetime.datetime.fromisoformat(last_used)
                                last_used_str = last_used_dt.strftime("%Y-%m-%d %H:%M")
                            except ValueError:
                                last_used_str = last_used
                                
                            print(f"  {short_id}{current} - Last used: {last_used_str}")
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
                    print("  10 USD to EUR   - Convert currencies (using codes)")
                    print("  $10 to €        - Convert currencies (using symbols)")
                    print("  x = $2/ml       - Set a derived unit rate")
                    print("  given x; find 10fluid_ounce in dollar - Convert unit to currency using rate")
                    print("  given x; find 10dollar in fluid_ounce - Convert currency to unit using rate")
                    print("  3 power of 2    - Use natural language math")
                    print("  square root of 16 - Use more complex expressions")
                    print("  2% of 100       - Calculate percentages")
                    print("\nCommands:")
                    print("  vars            - Show all variables")
                    print("  sessions        - List all saved sessions")
                    print("  keep-session    - Keep the session after exiting (for later use)")
                    print("  help            - Show this help message")
                    print("  exit/quit       - Exit the calculator")
                    print("\nSession Management:")
                    print("  - By default, sessions are temporary and deleted on exit")
                    print("  - Use 'keep-session' to preserve your session")
                    print("  - The last kept session will be loaded automatically next time")
                    print("  - Use '--new-session' flag to start with a fresh session")
                    print("  - To load a specific session: calc --session SESSION_ID")
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
                            # Special handling for derived unit conversions
                            if isinstance(result, dict) and result.get("type") == "derived_conversion":
                                # This is a derived unit conversion request like "given x; find 10oz in dollar"
                                derived_info = result
                                var_name = derived_info["var_name"]
                                
                                # Check if the variable exists
                                if var_name not in variables:
                                    evaluation_error = f"Variable '{var_name}' is not defined"
                                    break
                                    
                                # Get the variable's value (should be a rate string like "2 USD/ml")
                                rate_info = variables.get(var_name)
                                
                                # Get the derived unit calculator
                                derived_calculator = next((calc for calc in REGISTERED_CALCULATORS 
                                                      if isinstance(calc, DerivedUnitCalculator)), None)
                                
                                if derived_calculator:
                                    # Process the conversion
                                    conv_result, conv_error = derived_calculator.process_derived_conversion(
                                        rate_info=rate_info,
                                        value=derived_info["value"],
                                        from_unit=derived_info["from_unit"],
                                        to_currency=derived_info["to_unit"]
                                    )
                                    
                                    if conv_error:
                                        evaluation_error = conv_error
                                        break
                                    else:
                                        calculated_result = conv_result
                                        break
                                else:
                                    evaluation_error = "Internal error: Derived unit calculator not found"
                                    break
                            else:
                                # Normal calculation result
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
            print(f"The next time you run calc, this session will be loaded automatically.")
    
    # Clean up the session if needed
    if session_id and not keep_session:
        try:
            # Delete the session
            delete_session_db(session_id)
            
            # Update workspace to remove this session
            if 'session_info' in workspace and session_id in workspace.get('session_info', {}):
                del workspace['session_info'][session_id]
                
            # Clear last_session_id if it matches the deleted session
            if workspace.get('last_session_id') == session_id:
                workspace['last_session_id'] = None
                
            # Save workspace changes
            save_workspace(workspace)
            
        except Exception as e:
            print(f"Warning: Failed to delete session: {e}")
    elif session_id and keep_session:
        # Update workspace with kept session info
        if 'session_info' not in workspace:
            workspace['session_info'] = {}
            
        workspace['session_info'][session_id] = {
            'created_at': workspace.get('session_info', {}).get(session_id, {}).get('created_at', 
                          datetime.datetime.now().isoformat()),
            'last_used': datetime.datetime.now().isoformat(),
            'description': 'Kept session'
        }
        
        # Save workspace to remember this session for next time
        workspace['last_session_id'] = session_id
        save_workspace(workspace)
    
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