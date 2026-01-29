"""
Diagnostic script to check DynamicVis structure.
"""

import sys
from pathlib import Path

# Add DynamicVis to path
DYNAMICVIS_PATH = Path(__file__).parent / "architectures" / "DynamicVis"
sys.path.insert(0, str(DYNAMICVIS_PATH))

print("=" * 60)
print("DynamicVis Diagnostic")
print("=" * 60)

# Check directory structure
print("\n1. Checking DynamicVis directory structure...")
dynamicvis_dir = DYNAMICVIS_PATH / "dynamicvis"
if dynamicvis_dir.exists():
    print(f"   dynamicvis/ exists")
    for item in dynamicvis_dir.iterdir():
        print(f"   - {item.name}")
else:
    print(f"   dynamicvis/ NOT FOUND")

# Check models directory
print("\n2. Checking models directory...")
models_dir = dynamicvis_dir / "models"
if models_dir.exists():
    print(f"   models/ exists")
    for item in models_dir.iterdir():
        print(f"   - {item.name}")
else:
    print(f"   models/ NOT FOUND")

# Check what's in __init__.py
print("\n3. Checking models/__init__.py...")
init_file = models_dir / "__init__.py"
if init_file.exists():
    with open(init_file, "r") as f:
        content = f.read()
    print(f"   Contents:\n{content[:1000]}")
else:
    print("   __init__.py NOT FOUND")

# Try importing
print("\n4. Attempting imports...")
try:
    import dynamicvis
    print(f"   dynamicvis imported: {dir(dynamicvis)}")
except Exception as e:
    print(f"   Failed to import dynamicvis: {e}")

try:
    import dynamicvis.models
    print(f"   dynamicvis.models imported: {dir(dynamicvis.models)}")
except Exception as e:
    print(f"   Failed to import dynamicvis.models: {e}")

# Check MMPretrain registry
print("\n5. Checking MMPretrain registry...")
try:
    from mmengine.registry import MODELS
    
    # Import dynamicvis to register models
    try:
        import dynamicvis.models
    except:
        pass
    
    # Find relevant models
    relevant = [k for k in MODELS.module_dict.keys() 
                if any(x in k.lower() for x in ['dynamic', 'vis', 'vit'])]
    print(f"   Relevant registered models: {relevant[:20]}")
except Exception as e:
    print(f"   Could not check registry: {e}")

# Check config files
print("\n6. Checking fMoW config files...")
fmow_config_dir = DYNAMICVIS_PATH / "configs_DynamicVis" / "fMoW"
if fmow_config_dir.exists():
    for item in fmow_config_dir.iterdir():
        print(f"   - {item.name}")
        if item.suffix == ".py":
            with open(item, "r") as f:
                content = f.read()
            # Print first 50 lines
            lines = content.split("\n")[:50]
            print("     First 50 lines:")
            for line in lines:
                print(f"     {line}")
            print("     ...")
else:
    print("   fMoW config dir NOT FOUND")

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)