# Quick NLP Test for Earnings Report Analysis

This guide explains how to run quick tests on the NLP earnings report analysis with a reduced dataset size, allowing for rapid testing and debugging. The test scripts follow the same comprehensive Google-style documentation standards as the main codebase to ensure consistency and clarity.

## Running Quick Tests

We've created a special test file that allows you to run the full NLP pipeline on a small subset of the data, which runs much faster than the full 8500 text dataset.

### Using PowerShell

```powershell
# Change to the project directory
cd c:\Users\tcmk_\Downloads\NLP_earnings_report

# Run with minimal dataset (20 samples)
python tests\test_advanced_nlp_quick.py --sample-size 20 --max-features 100 --num-topics 2

# Run with slightly larger dataset
python tests\test_advanced_nlp_quick.py --sample-size 100 --max-features 500 --num-topics 5

# Run with medium dataset
python tests\test_advanced_nlp_quick.py --sample-size 500 --max-features 1000 --num-topics 10
```

## Parameters

You can customize the test run with these parameters:

- `--sample-size`: Number of samples to use (default: 100)
- `--max-features`: Maximum number of features for embeddings (default: 500)
- `--num-topics`: Number of topics for topic modeling (default: 5)

## Log Files

The test logs are saved to `advanced_nlp_quick_test.log` in the project root.

## When to Use Quick Tests vs. Full Tests

- **Quick Tests**: Use during development, debugging, and initial testing
- **Full Tests**: Use for final validation before deployment or when accuracy is critical

Once your quick tests pass successfully, you can run the full test with the original test script:

```powershell
python tests\test_advanced_nlp.py
```

## Troubleshooting

If you encounter errors, check:
1. The log file for detailed error information
2. Permission errors for feature extractor directories (common issue)
3. PyTorch/Streamlit compatibility issues (may need environment variable adjustment)
4. Ensure any alternative feature extractor paths are accessible

### Handling Common Errors

#### Feature Extractor Permission Issues

If you see errors like:
```
Failed to load feature extractor: [Errno 13] Permission denied: 'models/features/combined_features'
```

The system will automatically try alternative paths with timestamp suffixes:
```
Permission denied on original path, trying alternative: models/features/combined_features_1746907478
```

#### PyTorch/Streamlit Integration

If you encounter this error:
```
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
```

Make sure the environment variable is set correctly at the beginning of the dashboard code:
```python
os.environ["STREAMLIT_WATCH_MODULE_PATHS_EXCLUDE"] = "torch,torchaudio,torchvision,pytorch_pretrained_bert,transformers"
```

## Documentation References

All test scripts follow the same Google-style docstring standard as the main codebase:
- One-line summary description
- Detailed multi-line explanation
- Args section with parameter types and descriptions
- Returns section with return value types and descriptions
- Examples section showing typical usage patterns
2. File permissions if you encounter errors related to feature extractors
3. Environment variables if PyTorch and Streamlit integration shows errors

### Known Issues and Solutions

#### PyTorch and Streamlit Integration

The test may show the following error related to PyTorch classes:
```
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
```

**Solution**: This is caused by the Streamlit file watcher trying to introspect PyTorch classes. The code already includes the fix by setting an environment variable:

```powershell
$env:STREAMLIT_WATCH_MODULE_PATHS_EXCLUDE = "torch,torchaudio,torchvision,pytorch_pretrained_bert,transformers"
```

This is automatically set in the dashboard application.

#### Feature Extractor Permission Issues

You may encounter permission errors like:
```
Permission denied: 'models/features/combined_features'
```

**Solution**: The code handles this by creating alternative paths with timestamps appended:
```
Permission denied on original path, trying alternative: models/features/combined_features_[timestamp]
```

However, when loading these models, you might need to manually specify the alternative path if the dashboard doesn't automatically find it.
2. That the data pipeline has been run and data files exist
3. That all dependencies are installed correctly
4. Check permissions on model directories (especially for feature extractors)
5. Restart Streamlit with the `--server.fileWatcherType none` option if you encounter PyTorch integration errors

## How This Works

The quick test:
1. Loads only a small subset of the training, validation, and test data
2. Uses fewer features for embeddings
3. Uses fewer topics for topic modeling
4. Reduces iterations in machine learning models
5. Saves results in separate directories to avoid overwriting your full test results
