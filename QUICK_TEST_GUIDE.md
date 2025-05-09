# Quick NLP Test for Earnings Report Analysis

This guide explains how to run quick tests on the NLP earnings report analysis with a reduced dataset size, allowing for rapid testing and debugging.

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
