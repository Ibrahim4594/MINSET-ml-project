# Contributing to MNIST Digit Recognition

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/MNIST-ml-project.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `python -m pytest tests.py -v`
6. Commit with meaningful message
7. Push to your fork
8. Create a Pull Request

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests.py -v

# Run benchmarks
python benchmark.py

# Train model
python train_model.py

# Start development server
python app.py
```

## Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Keep functions focused and small
- Use meaningful variable names

## Commit Messages

Use the following format:

```
type: brief description

Longer description if needed. Explain the why, not the what.

- Use bullet points for multiple changes
- Reference issue numbers if applicable (#123)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation updates
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions/updates
- `chore`: Maintenance tasks

## Testing

- Add unit tests for new functionality
- Ensure all tests pass: `pytest tests.py -v`
- Aim for >80% code coverage
- Test edge cases and error conditions

## Documentation

- Update README.md if adding features
- Add docstrings in Google format
- Include usage examples
- Document API changes

## Performance

- Profile code before optimizing
- Use benchmarks to measure improvements
- Consider memory usage
- Document performance trade-offs

## Security

- Don't commit secrets or API keys
- Validate user input
- Use secure dependencies
- Report security issues privately

## Questions?

- Open an issue for discussions
- Check existing issues/PRs first
- Be respectful and constructive

Happy contributing!
