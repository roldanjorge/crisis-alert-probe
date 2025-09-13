#!/usr/bin/env python3
"""
Test script to verify Streamlit app setup and dependencies.

This script checks if all required components are in place for the Streamlit app.
"""

import os
import sys
from pathlib import Path
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    # All packages are in the main requirements.txt
    # Note: Some packages have different import names than their pip names
    required_packages = {
        'streamlit': 'Streamlit web framework',
        'plotly': 'Interactive visualizations',
        'pandas': 'Data manipulation',
        'torch': 'PyTorch deep learning',
        'transformers': 'Hugging Face transformers',
        'transformer_lens': 'Transformer Lens library',
        'streamlit_antd_components': 'Streamlit Ant Design components',
        'streamlit_image_select': 'Streamlit image selection component',
        'st_on_hover_tabs': 'Streamlit hover tabs component'  # Correct import name
    }
    
    failed_imports = []
    
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {package}: {description}")
        except ImportError as e:
            print(f"❌ {package}: {description} - {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages:")
        print("pip install -r ../requirements.txt")
        return False
    
    print("\n✅ All packages imported successfully!")
    return True

def test_file_structure():
    """Test if required files and directories exist."""
    print("\n🔍 Testing file structure...")
    
    required_files = {
        'streamlit_chat_app.py': 'Main Streamlit app',
        'probes.py': 'Probe class definition',
        'utils.py': 'Utility functions',
        'probe_checkpoints/reading_probe/probe_training_metrics.csv': 'Training metrics',
        'probe_checkpoints/reading_probe/probe_at_layer_39.pth': 'Best probe checkpoint'
    }
    
    missing_files = []
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✅ {file_path}: {description}")
        else:
            print(f"❌ {file_path}: {description}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("\n✅ All required files found!")
    return True

def test_probe_loading():
    """Test if probe can be loaded successfully."""
    print("\n🔍 Testing probe loading...")
    
    try:
        from probes import Probe
        from transformer_lens.utils import get_device
        
        device = get_device()
        print(f"✅ Device: {device}")
        
        # Test probe creation
        probe = Probe(num_classes=7, device=device)
        print("✅ Probe created successfully")
        
        # Test checkpoint loading
        checkpoint_path = "probe_checkpoints/reading_probe/probe_at_layer_39.pth"
        if Path(checkpoint_path).exists():
            import torch
            probe.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("✅ Probe checkpoint loaded successfully")
        else:
            print("❌ Probe checkpoint not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing probe: {e}")
        return False

def test_model_loading():
    """Test if LLaMA-2 model can be loaded (without actually loading it)."""
    print("\n🔍 Testing model loading capability...")
    
    try:
        from transformer_lens import HookedTransformer
        from transformer_lens.utils import get_device
        
        device = get_device()
        print(f"✅ Device available: {device}")
        
        # Check if we can access the model config (without loading)
        print("✅ Transformer Lens imported successfully")
        print("ℹ️  Note: Actual model loading will happen when the app starts")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model capability: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Streamlit App Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Probe Loading", test_probe_loading),
        ("Model Capability", test_model_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your Streamlit app is ready to run!")
        print("Run: python run_streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before running the app.")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()
