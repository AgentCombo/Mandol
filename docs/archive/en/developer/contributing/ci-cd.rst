CI/CD
=========

Automatically triggered on each push:

1. ``ruff check`` — Code style check
2. ``make test-unit`` — Unit tests
3. ``make test-integration`` — Integration tests (if API Key available)

Documentation is automatically generated and deployed after passing.
