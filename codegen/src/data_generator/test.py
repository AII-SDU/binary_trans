from binaryninja import *
import re
import os

def parse_instruction(ins, symbol_map, string_map):
    ins = re.sub(r'[^\x00-\x7F]+', '', ins)
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    if len(parts) > 1:
        operand = parts[1:]
    for i in range(len(operand)):
        symbols = re.split('([0-9A-Za-z]+)', operand[i])
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) >= 6:
                if int(symbols[j], 16) in symbol_map:
                    symbols[j] = "symbol" # function names
                elif int(symbols[j], 16) in string_map:
                    symbols[j] = "string" # constant strings
                elif symbols[j-1] != '#' and symbols[j][:3] !='0xf':
                    symbols[j] = "address" # addresses 
        operand[i] = ' '.join(symbols)
    opcode = parts[0]
    return ' '.join([opcode]+operand)

def process_file(file_path, output_path):
    symbol_map = {}
    string_map = {}
    print(file_path)
    bv = BinaryViewType.get_view_of_file(file_path)

    # encode strings
    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value

    disassembly = ""
    for func in bv.functions:
        if func.basic_blocks:  
            for block in func:
                for line in block.disassembly_text:
                    normalized_instruction = parse_instruction(bv.get_disassembly(line.address), symbol_map, string_map)
                    disassembly += f"{hex(line.address)}: {normalized_instruction}\n"
        # for block in function:
        #     for line in block.disassembly_text:
        #         raw_instruction = ''.join([token.text for token in line.tokens])
        #         normalized_instruction = parse_instruction(raw_instruction, symbol_map, string_map)
        #         disassembly += f"{hex(line.address)}: {normalized_instruction}\n"

    with open(output_path, 'a') as f:
        f.write(disassembly)

def process_string(f):
    str_lst = [] 
    bv = BinaryViewType.get_view_of_file(f)
    for sym in bv.get_symbols():
        str_lst.extend(re.findall('([0-9A-Za-z]+)', sym.full_name))
    return str_lst

def main():
    # 使用相对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # 假设数据集在项目目录中的dataset子目录
    dataset_dir = os.path.join(project_dir, 'dataset', 'binutils-2.35-x86-clang', 'dump')
    bin_path = os.path.join(dataset_dir, 'addr2line')
    output_path = os.path.join(dataset_dir, 'addr2line-x86-clang-ninja')
    
    process_file(bin_path, output_path)

if __name__ == "__main__":
    main()
