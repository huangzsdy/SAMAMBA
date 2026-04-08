# test_import.py
try:
    from basicseg.networks.sam2.modeling.sam2_base import SAM2Base
    from basicseg.networks.sam2.modeling import sam2_base
    print("✅ 导入成功")
    print(f"SAM2Base:{sam2_base.__file__}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    
    # 检查模块结构
    import sys
    import os
    
    # 打印 Python 路径
    print("\nPython 路径:")
    for path in sys.path:
        print(f"  {path}")
    
    # 检查 basicseg 是否存在
    try:
        import basicseg
        print(f"\nbasicseg 位置: {basicseg.__file__}")
        
        # 尝试列出 networks
        import basicseg.networks
        print(f"networks 位置: {basicseg.networks.__file__}")
    except ImportError as e2:
        print(f"\nbasicseg 导入失败: {e2}")
