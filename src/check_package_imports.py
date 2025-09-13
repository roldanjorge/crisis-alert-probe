#!/usr/bin/env python3
"""
Script to check the actual import names for Streamlit packages.
This helps identify the correct import names vs pip package names.
"""

def test_import_name(pip_name, possible_import_names):
    """Test different possible import names for a package."""
    print(f"Testing package: {pip_name}")
    
    for import_name in possible_import_names:
        try:
            __import__(import_name)
            print(f"✅ {pip_name} -> import as '{import_name}'")
            return import_name
        except ImportError:
            print(f"❌ {pip_name} -> '{import_name}' failed")
    
    print(f"❌ {pip_name} -> No working import name found")
    return None

def main():
    """Test all Streamlit packages."""
    print("🔍 Checking Streamlit package import names...")
    print("=" * 60)
    
    # Test packages with their possible import names
    packages_to_test = [
        ("streamlit-antd-components", ["streamlit_antd_components", "streamlit_antd", "antd_components"]),
        ("streamlit-image-select", ["streamlit_image_select", "image_select", "streamlit_image"]),
        ("streamlit-on-Hover-tabs", ["streamlit_on_hover_tabs", "streamlit_on_Hover_tabs", "hover_tabs", "streamlit_hover"])
    ]
    
    working_imports = {}
    
    for pip_name, import_names in packages_to_test:
        working_name = test_import_name(pip_name, import_names)
        if working_name:
            working_imports[pip_name] = working_name
        print()
    
    print("=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    if working_imports:
        print("✅ Working imports:")
        for pip_name, import_name in working_imports.items():
            print(f"  {pip_name} -> {import_name}")
    else:
        print("❌ No working imports found")
    
    print("\n💡 If packages are missing, install them with:")
    print("pip install streamlit-antd-components streamlit-image-select streamlit-on-Hover-tabs")

if __name__ == "__main__":
    main()
