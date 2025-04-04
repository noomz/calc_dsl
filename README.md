# Calculator DSL

A domain-specific language for performing various calculation tasks, including:

- Time calculations with flexible time format support
- Unit conversions
- Currency conversions
- Derived unit calculations (e.g., pricing rates like $2/ml)
- Mathematical operations with natural language support
- Variables and persistence
- Multiple interfaces (CLI, Web API)

## Features

### Time Calculator
- Calculate time differences with flexible input formats
- Support for various output formats (human-readable, 24-hour, AM/PM, seconds, etc.)
- Example: `9am - 5pm in minutes`

### Unit Calculator
- Convert between different units of measurement
- Example: `5 km to miles`

### Currency Calculator
- Convert between different currencies using up-to-date exchange rates
- Support for both currency codes and symbols
- Examples:
  - `10 USD to EUR`
  - `$10 to €`
  - `£25 to JPY`
  - `50 EUR to $`

### Derived Unit Calculator
- Create and use rates that combine currencies and physical units
- Useful for pricing, rates, and unit economics calculations
- Examples:
  - `x = $2/ml` (define a rate of $2 per milliliter)
  - `given x; find 10oz in dollar` (convert 10 ounces to dollars using rate x)

### Math Calculator
- Support for traditional syntax: `5 + 3`, `7 * 8`
- Natural language math expressions: 
  - `3 power of 10`
  - `square root of 9`
  - `2% of 100`
  - `factorial of 5`

### Persistent Variables
- Assign calculation results to variables: `x = 5 + 3`
- Use variables in subsequent calculations: `y = x * 2`

## Usage

### Command Line Interface (CLI)

The calculator provides an interactive command line interface with tab completion:

```bash
# Start the interactive calculator CLI
poetry run calc

# Example session:
calc> 2 + 2
Result: 4
calc> x = 10
x = 10
calc> x * 2
Result: 20
calc> 9am - 5pm in minutes
Result: 480
calc> vars
Current variables:
  x = 10
calc> help
# Shows available commands and examples
```

Tab completion makes it easy to:
- Complete commands: `h<TAB>` → `help`
- Access variables: `?x<TAB>` → `?x` (prints the value of x)
- Complete unit conversions: `5km to <TAB>` → shows available units
- Complete derived unit queries: `given x; <TAB>` → expands to `given x; find `
- Get example assignments: `y = <TAB>` → shows assignment examples

#### Session Management

The CLI has automatic session management that remembers your last session:

```bash
# Your session is automatically saved in ~/.config/calc_dsl/workspace.json
# By default, sessions are temporary and deleted on exit
# To keep a session for later use:
calc> keep-session
Session 123e4567-e89b-12d3-a456-426614174000 will be kept after exiting.
The next time you run calc, this session will be loaded automatically.

# Exit the calculator (session is preserved)
calc> exit

# Next time you run calc, your session is automatically loaded
poetry run calc
Loaded existing session: 123e4567-e89b-12d3-a456-426614174000

# View all your saved sessions
calc> sessions
Available sessions:
  123e4567... (current) - Last used: 2023-08-15 14:30

# Start with a fresh session instead of the last one
poetry run calc --new-session

# Load a specific session explicitly
poetry run calc --session 123e4567-e89b-12d3-a456-426614174000

# Start the CLI with a session that will be kept after exiting
poetry run calc --keep-session
```

This allows you to:
- Automatically continue where you left off
- Save your work between CLI sessions
- Manage multiple sessions
- Share sessions between CLI and API modes

### Web Server

The calculator also provides a web API:

```bash
# Start the web server
poetry run server
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/calc_dsl.git
cd calc_dsl

# Install with Poetry
poetry install

# Run the CLI
poetry run calc

# Run the web server
poetry run server
```

## API Endpoints

- `POST /calculate`: Perform a quick calculation without a session
- `POST /sessions`: Create a new calculation session
- `GET /sessions/<session_id>`: Get session variables
- `DELETE /sessions/<session_id>`: Delete a session
- `POST /sessions/<session_id>/calculate`: Perform calculation in session

## Examples

### Time Calculations
```
9am - 5pm                  → 8 hours
9:30 - 10:45 in minutes    → 75
1pm - 4:30pm in seconds    → 12600
```

### Unit Conversions
```
5 km to miles              → 3.10686 miles
100 kg to pounds           → 220.462 pounds
30 celsius to fahrenheit   → 86 °F
```

### Currency Conversions
```
10 USD to EUR              → 8.50 EUR
$10 to €                   → 8.50 EUR
£25 to JPY                 → 3600.00 JPY
50 EUR to $                → 58.82 USD
```

### Derived Unit Conversions
```
x = $2/ml                  → 2 USD/ml
given x; find 10fluid_ounce in dollar → 590.59 USD  (10 fluid ounces at $2/ml)
y = 5 EUR/km               → 5 EUR/km
given y; find 10miles in euro → 80.47 EUR  (10 miles at 5 EUR/km)
```

### Math Expressions
```
3 power of 2               → 9
square root of 16          → 4
2% of 100                  → 2
factorial of 5             → 120
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.