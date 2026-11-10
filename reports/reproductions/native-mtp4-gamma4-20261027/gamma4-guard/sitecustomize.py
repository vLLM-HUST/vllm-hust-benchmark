import os
import sys
import traceback
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
try:
    exec(compile((root/'guard/sitecustomize.py').read_text(),str(root/'guard/sitecustomize.py'),'exec'),{'__file__':str(root/'guard/sitecustomize.py')})
    from gamma4_witness import install
    install()
except BaseException:
    traceback.print_exc()
    os._exit(78)
