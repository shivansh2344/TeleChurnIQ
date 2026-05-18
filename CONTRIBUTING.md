# Contributing to TeleChurnIQ

Thank you for your interest in contributing to TeleChurnIQ! We welcome contributions from the community.

## Code of Conduct

Please treat all members with respect. Be constructive and helpful in discussions.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Provide a clear description of the bug
3. Include steps to reproduce
4. Share any error messages or logs
5. Specify your environment (OS, Python version, etc.)

### Suggesting Enhancements

1. Open an issue with a clear title
2. Describe the enhancement and why it's needed
3. Provide example use cases
4. Reference related issues if applicable

### Submitting Code Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/TeleChurnIQ.git
   cd TeleChurnIQ
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow PEP 8 style guide for Python
   - Use meaningful variable and function names
   - Add docstrings to functions and modules
   - Include comments for complex logic

4. **Test your changes**
   ```bash
   # Python tests
   pytest

   # Frontend tests (when available)
   cd frontend && npm test
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: Description of changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots for UI changes
   - Request review from maintainers

## Development Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/TeleChurnIQ.git
cd TeleChurnIQ

# Setup Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install backend dependencies
cd backend
npm install
cd ..

# Install frontend dependencies
cd frontend
npm install
cd ..
```

## Code Style Guidelines

### Python
- Follow PEP 8 standards
- Use type hints where applicable
- Maximum line length: 100 characters
- Use descriptive variable names

```python
# Good
def calculate_churn_probability(customer_features: dict) -> float:
    """Calculate churn probability for a customer."""
    pass

# Avoid
def calc(x):
    pass
```

### JavaScript/React
- Use arrow functions and functional components
- Use meaningful component names
- Add prop validation
- Keep components under 300 lines

```javascript
// Good
const ChurnPredictionCard = ({ prediction, probability }) => {
  return <div>{probability}%</div>;
};

// Avoid
const Card = (props) => {
  return <div>{props.p}%</div>;
};
```

### Commit Messages
- Use present tense: "Add feature" not "Added feature"
- Be specific: "Add SHAP explainability" not "Fix stuff"
- Reference issues: "Closes #123"

```
Add feature: SHAP explainability engine

- Implement SHAP TreeExplainer integration
- Add narrative generation for explanations
- Update prediction API with explanation output

Closes #123
```

## Pull Request Process

1. Ensure tests pass locally
2. Update README if adding new features
3. Add docstrings to new functions
4. Request review from team members
5. Address review comments
6. Rebase before merging

## Project Structure

```
TeleChurnIQ/
├── frontend/          # React UI
├── backend/           # Express API
├── ml-service/        # Python ML Pipeline
├── model_experiments/ # Research & Training
└── README.md          # Project documentation
```

## Questions?

- Open an issue on GitHub
- Check existing discussions
- Review the main README for architecture overview

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to TeleChurnIQ! 🚀
