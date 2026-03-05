import os
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import StructuredTool
from .base_tool import BaseToolService

class PythonChartService(BaseToolService):
    def __init__(self, save_dir: str = "static"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.repl = PythonREPL()

    def python_chart_maker(self, code: str) -> str:
        print("[Chart Tool] Executing Python code...")
        
        wrapped_code = f"""
import matplotlib.pyplot as plt
import pandas as pd
import os

# Code của AI bắt đầu
{code}
# Code của AI kết thúc

# Force save file
if plt.get_fignums():
    save_path = '{self.save_dir}/chart_output.png'
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"SUCCESSS_CHART_SAVED: {{save_path}}")
else:
    print("NO_CHART_CREATED")
"""
        try:
            result = self.repl.run(wrapped_code)
            
            if "SUCCESSS_CHART_SAVED" in result:
                return f"Đã vẽ biểu đồ thành công và lưu tại '{self.save_dir}/chart_output.png'. Hãy hiển thị nó cho người dùng."
            elif "NO_CHART_CREATED" in result:
                return f"Code đã chạy nhưng không tạo ra biểu đồ. Output: {result}"
            else:
                return f"Kết quả chạy code: {result}"
                
        except Exception as e:
            return f"Lỗi Python: {str(e)}"

    def get_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.python_chart_maker,
            name="python_chart_maker",
            description=(
                "Công cụ chạy code Python để phân tích dữ liệu hoặc vẽ biểu đồ. "
                "Sử dụng thư viện: matplotlib, pandas. "
                "Input: Đoạn code Python hợp lệ. "
                "Output: Kết quả chạy code hoặc thông báo đã lưu ảnh."
            )
        )