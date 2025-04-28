import re

# 示例字符串
example_str = "mov w3  #0x1                    // #1"

# 正则表达式删除注释
cleaned_str = re.sub(r'\s*//.*$', '', example_str)

print("处理后的字符串：", cleaned_str,1)
print("处理后的字符串：", cleaned_str)