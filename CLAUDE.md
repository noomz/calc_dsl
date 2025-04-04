# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install: `poetry install`
- Run CLI: `poetry run calc`
- Run Web Server: `poetry run server`
- Format code: `poetry run black .`
- Sort imports: `poetry run isort .`
- Lint: `poetry run flake8 .`
- Type check: `poetry run mypy .`
- Run tests: `poetry run pytest`
- Run single test: `poetry run pytest path/to/test.py::test_name`

## Code Style Guidelines
- **Formatting**: Follow Black (line length 88) and isort with Black profile
- **Typing**: Use type hints for all function parameters and return values
- **Imports**: Group standard library, third-party, and local imports (isort)
- **Naming**: 
  - snake_case for variables, functions, methods
  - CamelCase for classes
  - ALL_CAPS for constants
- **Error handling**: Use try/except with specific exceptions and error logging
- **Documentation**: Docstrings for modules, classes, and functions
- **Variables**: Prefix internal variables with underscore (_var_name)
- **Patterns**: Use ABC for interfaces, regex for parsing user input