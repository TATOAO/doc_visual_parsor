#!/usr/bin/env python3
"""
Test runner script for Document Visual Parser backend tests.

This script provides an easy way to run all tests with various options.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(coverage=False, verbose=False, specific_test=None, parallel=False):
    """Run tests with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    cmd.append("tests/")
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=backend",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])
    
    # Add verbose output if requested
    if verbose:
        cmd.append("-v")
    
    # Add parallel execution if requested
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add specific test if provided
    if specific_test:
        cmd.append(specific_test)
    
    print(f"🧪 Running tests: {' '.join(cmd)}")
    print("─" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        
        if coverage:
            print("📊 Coverage report generated in htmlcov/ directory")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run backend tests for Document Visual Parser")
    parser.add_argument(
        "--coverage", "-c", 
        action="store_true", 
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Run tests with verbose output"
    )
    parser.add_argument(
        "--parallel", "-p", 
        action="store_true", 
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--test", "-t", 
        type=str, 
        help="Run specific test file or test function"
    )
    parser.add_argument(
        "--install-deps", 
        action="store_true", 
        help="Install test dependencies before running tests"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("📦 Installing test dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"
            ], check=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    # Check if backend directory exists
    backend_path = Path("backend")
    if not backend_path.exists():
        print("❌ Backend directory not found. Please run from project root.")
        return False
    
    # Check if tests directory exists
    tests_path = Path("tests")
    if not tests_path.exists():
        print("❌ Tests directory not found. Please run from project root.")
        return False
    
    # Run tests
    success = run_tests(
        coverage=args.coverage,
        verbose=args.verbose,
        specific_test=args.test,
        parallel=args.parallel
    )
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 