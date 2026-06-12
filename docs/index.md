# AutoBacktest Documentation

Welcome to the AutoBacktest documentation. This hub provides a structured entry point to all project documentation.

---

## Table of Contents

### Getting Started
- **[About Project](about-project.md)** — Business goals, target users, and primary interaction flows
- **[Developer Setup](developer-setup.md)** — Installation, environment configuration, and development commands
- **[Contributing](../CONTRIBUTING.md)** — Guidelines for contributing to the project

### Strategy Authoring
- **[Strategy Guide](strategy-guide.md)** — Complete guide to creating, configuring, and optimizing strategies

### Architecture & Design
- **[System Architecture](architecture.md)** — Module dependencies, component diagrams, and gate system design
- **[Optimization Config Reference](optimization-config-reference.md)** — Complete catalog of all configurable parameters

### API & Reference
- **[API Reference](api-reference.md)** — Typed documentation of all public endpoints, dataclasses, and utility libraries

### Quick Links
- **[Root README](../README.md)** — Project overview, quickstart, and CLI reference
- **[AGENTS.md](../AGENTS.md)** — Agent-facing commands, architecture notes, and testing quirks

---

## Documentation Map

```
AutoBacktest
├── README.md                          # Quickstart & CLI reference
├── AGENTS.md                          # Agent commands & architecture notes
├── CONTRIBUTING.md                    # Contribution guidelines
├── docs/
│   ├── index.md                       # This file — documentation hub
│   ├── about-project.md               # Business goals & user persona
│   ├── strategy-guide.md              # Strategy authoring guide
│   ├── architecture.md                # System architecture & module design
│   ├── developer-setup.md             # Setup, testing, and development workflow
│   ├── api-reference.md               # Public API documentation
│   └── optimization-config-reference.md  # Configuration parameter catalog
├── strategies/                        # Strategy subdirectories (<name>/strategy.py + config.yaml)
├── src/autobacktest/                  # Core engine source code
└── tests/                             # Test suite
```
