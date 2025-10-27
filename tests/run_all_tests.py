"""
Test runner script for the comprehensive test suite.
Runs all unit tests and integration tests with proper reporting.
"""

import unittest
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')

def run_all_tests():
    """Run all tests in the test suite."""
    print("=" * 80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILED TESTS:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERROR TESTS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Determine success
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        return 1

if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)