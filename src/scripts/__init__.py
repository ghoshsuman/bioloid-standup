import os
import sys

working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(working_dir)
print(sys.path)
os.chdir(working_dir)
