# PyPI Release Guide for QuickInsights

This guide explains how to publish QuickInsights to PyPI (Python Package Index).

## Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org)
2. **Test PyPI Account**: Create an account at [test.pypi.org](https://test.pypi.org)
3. **API Tokens**: Generate API tokens for both PyPI and Test PyPI

## Setup

### 1. Create API Tokens

#### For PyPI (Production):
1. Go to [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
2. Click "Add API token"
3. Give it a name (e.g., "QuickInsights v0.3.0")
4. Set scope to "Entire account" (for first upload) or "Specific project" (for updates)
5. Copy the token (starts with `pypi-`)

#### For Test PyPI:
1. Go to [test.pypi.org/manage/account/token/](https://test.pypi.org/manage/account/token/)
2. Follow the same steps as above
3. Copy the test token (starts with `pypi-`)

### 2. Configure .pypirc

1. Copy the template:
   ```bash
   cp .pypirc.template .pypirc
   ```

2. Edit `.pypirc` and replace the tokens:
   ```ini
   [pypi]
   username = __token__
   password = pypi-your-actual-production-token-here

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-your-actual-test-token-here
   ```

3. **Important**: Never commit `.pypirc` to version control!

## Publishing Process

### Method 1: Using the Automated Script (Recommended)

1. **Run the publishing script**:
   ```bash
   python publish_to_pypi.py
   ```

2. **Follow the prompts**:
   - Choose option 1 for Test PyPI (recommended for first upload)
   - Choose option 2 for production PyPI
   - Choose option 3 to build only (no upload)

### Method 2: Manual Publishing

#### Step 1: Clean and Build
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Install build tools
pip install --upgrade build twine

# Build the package
python -m build
```

#### Step 2: Check the Package
```bash
# Check the built package
python -m twine check dist/*
```

#### Step 3: Upload to Test PyPI (Recommended First)
```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*
```

#### Step 4: Test Installation from Test PyPI
```bash
# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ quickinsights

# Test the installation
python -c "import quickinsights; print(quickinsights.__version__)"
```

#### Step 5: Upload to Production PyPI
```bash
# Upload to production PyPI
python -m twine upload dist/*
```

## Verification

### After Upload to Test PyPI:
- Visit: https://test.pypi.org/project/quickinsights/
- Test installation: `pip install --index-url https://test.pypi.org/simple/ quickinsights`

### After Upload to Production PyPI:
- Visit: https://pypi.org/project/quickinsights/
- Test installation: `pip install quickinsights`

## Package Information

The package includes:
- **Source code**: All Python modules in `src/quickinsights/`
- **Documentation**: README.md, API reference, examples
- **Configuration**: pyproject.toml, requirements.txt
- **License**: MIT License
- **Metadata**: Comprehensive classifiers and keywords

## Troubleshooting

### Common Issues:

1. **"Package already exists"**:
   - Increment version in `pyproject.toml`
   - Update `__version__` in `src/quickinsights/__init__.py`

2. **"Invalid credentials"**:
   - Check your `.pypirc` file
   - Verify API tokens are correct
   - Ensure tokens have proper permissions

3. **"Package check failed"**:
   - Run `python -m twine check dist/*` to see specific errors
   - Check for missing files in MANIFEST.in
   - Verify all dependencies are properly specified

4. **"Build failed"**:
   - Ensure all dependencies are installed
   - Check for syntax errors in source code
   - Verify pyproject.toml is valid

### Version Management:

- **Major version** (1.0.0): Breaking changes
- **Minor version** (0.4.0): New features, backward compatible
- **Patch version** (0.3.1): Bug fixes, backward compatible

## Security Notes

- **Never commit** `.pypirc` to version control
- **Use API tokens** instead of passwords
- **Test on Test PyPI** before production upload
- **Keep tokens secure** and rotate them regularly

## Post-Release

After successful upload:

1. **Update documentation** with new version
2. **Create GitHub release** with changelog
3. **Announce** on social media/forums
4. **Monitor** for issues and feedback
5. **Plan** next version features

## Support

If you encounter issues:
- Check PyPI documentation: https://packaging.python.org/
- Review error messages carefully
- Test with Test PyPI first
- Contact PyPI support if needed

---

**Happy Publishing!** 🚀
