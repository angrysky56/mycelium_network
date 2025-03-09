#!/usr/bin/env python3
"""
Health check script for the mycelium network project.

This script checks for common issues and verifies that the environment
is set up correctly.
"""

import os
import sys
import importlib
import subprocess
import platform
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    required_version = (3, 7)
    current_version = sys.version_info[:2]
    
    return current_version >= required_version


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are installed."""
    required_packages = [
        "numpy",
        "matplotlib",
        "sklearn",  # Using sklearn instead of scikit-learn for import check
        "pytest",
        "jupyterlab",
        "plotly",
        "seaborn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages


def check_import_paths() -> Tuple[bool, List[str]]:
    """Check if the import paths are correct."""
    import_checks = [
        "from mycelium.environment import Environment",
        "from mycelium.network import AdvancedMyceliumNetwork",
        "from mycelium.enhanced.rich_environment import RichEnvironment",
        "from mycelium.enhanced.adaptive_network import AdaptiveMyceliumNetwork",
        "from mycelium.enhanced.resource import ResourceType",
        "from mycelium.optimized.batch_network import BatchProcessingNetwork",
    ]
    
    failed_imports = []
    
    for import_stmt in import_checks:
        try:
            exec(import_stmt)
        except ImportError as e:
            failed_imports.append(f"{import_stmt} -> {str(e)}")
    
    return len(failed_imports) == 0, failed_imports


def check_tests() -> Tuple[bool, str]:
    """Run a quick test to check if the core functionality works."""
    try:
        # Run the quick test script
        process = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(os.path.dirname(__file__)), "quick_test.py")],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if process.returncode == 0 and "Test completed successfully" in process.stdout:
            return True, "Quick test passed"
        else:
            return False, f"Quick test failed with output:\n{process.stdout}\n{process.stderr}"
    except Exception as e:
        return False, f"Error running quick test: {str(e)}"


def check_environment() -> Dict[str, Any]:
    """Check the overall environment and return results."""
    results = {}
    
    # Check Python version
    python_version_ok = check_python_version()
    results["python_version"] = {
        "status": "PASS" if python_version_ok else "FAIL",
        "details": f"Python {sys.version}"
    }
    
    # Check virtual environment
    in_venv = hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    results["virtual_env"] = {
        "status": "PASS" if in_venv else "WARN",
        "details": "Virtual environment detected" if in_venv else "Not running in a virtual environment"
    }
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    results["dependencies"] = {
        "status": "PASS" if deps_ok else "FAIL",
        "details": "All dependencies installed" if deps_ok else f"Missing dependencies: {', '.join(missing_deps)}"
    }
    
    # Check import paths
    imports_ok, failed_imports = check_import_paths()
    results["imports"] = {
        "status": "PASS" if imports_ok else "FAIL",
        "details": "All imports successful" if imports_ok else f"Failed imports:\n  " + "\n  ".join(failed_imports)
    }
    
    # Check tests
    tests_ok, test_message = check_tests()
    results["tests"] = {
        "status": "PASS" if tests_ok else "FAIL",
        "details": test_message
    }
    
    # Overall status
    all_pass = all(result["status"] == "PASS" for result in results.values())
    warn_only = not all_pass and all(result["status"] in ["PASS", "WARN"] for result in results.values())
    
    if all_pass:
        results["overall"] = {
            "status": "PASS",
            "details": "All checks passed. Your environment is set up correctly."
        }
    elif warn_only:
        results["overall"] = {
            "status": "WARN",
            "details": "Some checks generated warnings. Your environment may have issues."
        }
    else:
        results["overall"] = {
            "status": "FAIL",
            "details": "Some checks failed. Your environment is not correctly set up."
        }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print the results of the health check in a nice format."""
    print("Mycelium Network Project Health Check")
    print("====================================\n")
    
    # Get terminal width
    try:
        terminal_width = os.get_terminal_size().columns
    except:
        terminal_width = 80
    
    # Print environment info
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Path: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    print("-" * terminal_width + "\n")
    
    # Print individual check results
    for check, result in results.items():
        if check == "overall":
            continue
            
        status = result["status"]
        details = result["details"]
        
        status_color = ""
        reset_color = ""
        
        # Use colors if supported
        if sys.stdout.isatty():
            if status == "PASS":
                status_color = "\033[92m"  # Green
            elif status == "WARN":
                status_color = "\033[93m"  # Yellow
            elif status == "FAIL":
                status_color = "\033[91m"  # Red
                
            reset_color = "\033[0m"  # Reset
        
        # Print check result
        print(f"{check.replace('_', ' ').title()}:")
        print(f"  Status: {status_color}{status}{reset_color}")
        print(f"  Details: {details}")
        print()
    
    # Print overall result
    overall = results["overall"]
    status = overall["status"]
    details = overall["details"]
    
    status_color = ""
    reset_color = ""
    
    # Use colors if supported
    if sys.stdout.isatty():
        if status == "PASS":
            status_color = "\033[92m"  # Green
        elif status == "WARN":
            status_color = "\033[93m"  # Yellow
        elif status == "FAIL":
            status_color = "\033[91m"  # Red
            
        reset_color = "\033[0m"  # Reset
    
    print("-" * terminal_width)
    print(f"Overall: {status_color}{status}{reset_color}")
    print(details)


def provide_solutions(results: Dict[str, Any]) -> None:
    """Provide possible solutions for failed checks."""
    if results["overall"]["status"] == "PASS":
        return
    
    print("\nPossible Solutions:")
    print("------------------\n")
    
    # Python version
    if results["python_version"]["status"] == "FAIL":
        print("Python Version:")
        print("  - Install Python 3.7 or newer")
        print("  - Use a tool like pyenv to manage multiple Python versions")
        print()
    
    # Virtual environment
    if results["virtual_env"]["status"] == "WARN":
        print("Virtual Environment:")
        print("  - Create and activate a virtual environment:")
        print("    python -m venv venv")
        print("    source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print()
    
    # Dependencies
    if results["dependencies"]["status"] == "FAIL":
        print("Dependencies:")
        print("  - Install missing dependencies:")
        missing_deps = results["dependencies"]["details"].split(": ")[-1].split(", ")
        print(f"    pip install {' '.join(missing_deps)}")
        print("  - Or install all dependencies at once:")
        print("    pip install -r requirements.txt")
        print()
    
    # Import paths
    if results["imports"]["status"] == "FAIL":
        print("Import Paths:")
        print("  - Make sure you're running from the project root directory")
        print("  - Check that all modules are installed correctly")
        print("  - Common import path issues:")
        print("    - Use 'from mycelium.enhanced.rich_environment import RichEnvironment'")
        print("      (not 'from mycelium.enhanced.environment import RichEnvironment')")
        print()
    
    # Tests
    if results["tests"]["status"] == "FAIL":
        print("Tests:")
        print("  - Make sure all dependencies are installed")
        print("  - Check that you're in the virtual environment")
        print("  - Run the tests individually:")
        print("    python -m pytest tests/test_batch_network.py")
        print("    python -m pytest tests/test_genetic.py")
        print("    python -m pytest tests/test_spatial_index.py")
        print()


def main():
    """Run the health check and print results."""
    results = check_environment()
    print_results(results)
    
    # Only show solutions if there are issues
    if results["overall"]["status"] != "PASS":
        provide_solutions(results)
        
        # Exit with error code if there are failures
        if results["overall"]["status"] == "FAIL":
            sys.exit(1)
    
    # All pass
    if results["overall"]["status"] == "PASS":
        print("\nYour environment is ready for the Mycelium Network project!")
        print("Next steps:")
        print("  - Run example scripts: python examples/batch_processing_demo.py")
        print("  - Run all tests: python -m pytest tests/")
        print("  - Explore the documentation: cat README.md")


if __name__ == "__main__":
    main()
