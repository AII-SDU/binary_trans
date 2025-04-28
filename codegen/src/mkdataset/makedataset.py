import re
import csv
import os

def format_line_number_and_extract_assembly(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    parsed_data = {}
    current_line_number = None
    current_code_block = []
    is_gcc = 'gcc' in file_path

    for line in lines:
        line = line.strip()
        line_number_match = re.match(r'(.+:\d+)', line)

        if line_number_match:
            if current_line_number:
                parsed_data[current_line_number] = '\n'.join(current_code_block)
            full_path = line_number_match.group(1)
            filename_with_line = '/'.join(full_path.split('/')[-2:])
            current_line_number = filename_with_line
            current_code_block = []
        elif current_line_number:
            parts = line.split('\t')
            if (is_gcc and len(parts) >= 3) or (not is_gcc and len(parts) >= 2):
                asm_part = '\t'.join(parts[2:]) if is_gcc else '\t'.join(parts[1:])
                formatted_asm = format_assembly(asm_part)
                current_code_block.append(formatted_asm)

    if current_line_number and current_code_block:
        parsed_data[current_line_number] = '\n'.join(current_code_block)

    return parsed_data

def format_assembly(asm_code):
    asm_code = re.sub(r'[$%]', '', asm_code)  # Remove $ and % symbols
    asm_code = re.sub(r',', ' ', asm_code)  # Replace commas with spaces
    asm_code = re.sub(r'\s*//.*$', '', asm_code)
    # Additional formatting if '<' is present
    if '<' in asm_code:
        before_angle_bracket = re.search(r'^(.*?)(?:\s+<|$)', asm_code).group(1)
        parts = re.split(r'\s+|\t+', before_angle_bracket)
        if parts:
            parts.pop()  # Remove the last element
            reconstructed = ' '.join(parts)  # Reconstruct the remaining parts
            after_angle_bracket = re.search(r'<(.*)', asm_code).group(1)
            asm_code = reconstructed + ' <' + after_angle_bracket

    return asm_code.strip()

def write_mapping_to_csv(arm_data, x86_data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Source Line', 'ARM Assembly', 'x86 Assembly'])

        for line, asm in arm_data.items():
            if line in x86_data:
                writer.writerow([line, asm, x86_data[line]])


# 使用相对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 文件路径
arm_file_path = os.path.join(current_dir, 'addr2line-arm-gcc.txt')
x86_file_path = os.path.join(current_dir, 'addr2line-x86-gcc.txt')

# 处理文件
arm_data = format_line_number_and_extract_assembly(arm_file_path)
x86_data = format_line_number_and_extract_assembly(x86_file_path)

# 将映射写入CSV文件
output_csv = os.path.join(current_dir, 'assembly_mapping.csv')
write_mapping_to_csv(arm_data, x86_data, output_csv)
