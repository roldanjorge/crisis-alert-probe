#!/usr/bin/env python3
"""
Launcher script for the Streamlit Chat App with Emotional Monitoring.

This script sets up the environment and launches the Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    # Core packages (essential for the app)
    core_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'torch',
        'transformers',
        'transformer_lens'
    ]
    
    # Optional Streamlit packages (nice to have but not critical)
    optional_packages = [
        ('streamlit_antd_components', ['streamlit_antd_components', 'streamlit_antd']),
        ('streamlit_image_select', ['streamlit_image_select', 'image_select']),
        ('streamlit_on_hover_tabs', ['st_on_hover_tabs', 'streamlit_on_hover_tabs', 'streamlit_on_Hover_tabs'])
    ]
    
    missing_core = []
    missing_optional = []
    
    # Check core packages
    print("🔍 Checking core packages...")
    for package in core_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_core.append(package)
    
    # Check optional packages
    print("\n🔍 Checking optional packages...")
    for pip_name, import_names in optional_packages:
        found = False
        for import_name in import_names:
            try:
                __import__(import_name)
                print(f"✅ {pip_name} (imported as {import_name})")
                found = True
                break
            except ImportError:
                continue
        
        if not found:
            print(f"❌ {pip_name}")
            missing_optional.append(pip_name)
    
    # Report results
    if missing_core:
        print(f"\n❌ Missing core packages: {', '.join(missing_core)}")
        print("Please install them using:")
        print("pip install -r ../requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("The app will work without these, but some features may be limited.")
        print("To install them:")
        print("pip install streamlit-antd-components streamlit-image-select streamlit-on-Hover-tabs")
    
    print("\n✅ Core packages are available!")
    return True

def main():
    """Main launcher function."""
    print("🚀 Starting LLaMA-2 Chat App with Emotional Monitoring")
    print("=" * 60)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "streamlit_chat_app.py").exists():
        print("❌ Error: streamlit_chat_app.py not found in current directory")
        print(f"Current directory: {current_dir}")
        print("Please run this script from the src/ directory")
        return
    
    # Check requirements
    if not check_requirements():
        print("❌ Missing required packages. Please install them first.")
        return
    
    # Check if probe checkpoints exist
    checkpoint_dir = Path("probe_checkpoints/reading_probe")
    if not checkpoint_dir.exists():
        print("❌ Error: Probe checkpoints not found!")
        print(f"Expected directory: {checkpoint_dir.absolute()}")
        print("Please make sure you have trained probes in the correct location.")
        return
    
    # Check if metrics file exists
    metrics_file = checkpoint_dir / "probe_training_metrics.csv"
    if not metrics_file.exists():
        print("❌ Error: Training metrics file not found!")
        print(f"Expected file: {metrics_file.absolute()}")
        return
    
    print("✅ All requirements satisfied!")
    print("🌐 Launching Streamlit app...")
    print("📱 The app will open in your default web browser")
    print("🔄 If it doesn't open automatically, check the terminal for the URL")
    print("=" * 60)
    
    # Launch Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_chat_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main()
