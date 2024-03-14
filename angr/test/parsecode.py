import csv
import re
from keystone import *
amd64_registers = {
    # 64-bit registers
    'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
    'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
    # 32-bit registers
    'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
    'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
    # 16-bit registers
    'ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp',
    'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
    # 8-bit registers
    'al', 'bl', 'cl', 'dl', 'sil', 'dil', 'bpl', 'spl',
    'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b',
    'ah', 'bh', 'ch', 'dh'
}
aarch64_registers = {
    # 64-bit registers
    'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
    'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
    'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23',
    'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'sp', 'xzr',
    # 32-bit registers
    'w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7',
    'w8', 'w9', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15',
    'w16', 'w17', 'w18', 'w19', 'w20', 'w21', 'w22', 'w23',
    'w24', 'w25', 'w26', 'w27', 'w28', 'w29', 'w30', 'wsp'
}
def preprocess_instruction(instruction):
    instruction.rstrip()
    # 将换行符替换为分号
    while instruction.endswith('\n') or instruction.endswith(' '):
        instruction = instruction[:-1]  # 去除末尾的换行符
    instruction = instruction.replace("\n", ";")
    instruction = instruction.replace("endbr64 ;", "")
    instruction = instruction.replace("<unk> ", "")
    # 删除两个及以上连在一起的空格
    instruction = re.sub(r'\s{2,}', '', instruction)
    return instruction


def add_commas(insn, registers):
    parts = insn.split()
    if len(parts) <= 2:
        return insn  # 没有足够的操作数来添加逗号

    # 从第一个操作数到最后一个操作数添加逗号
    for i in range(1, len(parts) - 1):
        if parts[i] in registers or parts[i+1] in registers or '0x' in parts[i+1] and '[' in parts[i]:
            parts[i] = parts[i] + ','

    return ' '.join(parts)

def process_insn(insn, is_source=True):
    # 先添加逗号，然后进行其他处理
    
    
    # 处理 qword, dword, word, byte 添加 ptr
    for size in ["qword", "dword", "word", "byte"]:
        insn = insn.replace(f"{size} [", f"{size} ptr [")
    
    # 替换 symbol 和 address 为 label，考虑多个label的情况
    insn = insn.replace("symbol", "label").replace("address", "label").replace("string", "label")
    
    if is_source:
        # 对于 x86，删除 [ ] 中的 rel 和空格
        if '[' in insn and ']' in insn:
            start = insn.index('[')
            end = insn.index(']')
            inner_content = insn[start+1:end].replace(" rel ", "").replace(" ", "")
            insn = insn[:start+1] + inner_content + insn[end:]
    else:
        # 对于 ARM，删除 [ ] 中的所有内容
        if '[' in insn and ']' in insn:
            start = insn.index('[')
            end = insn.index(']')
            insn = insn[:start+1] + insn[end:]
    registers = amd64_registers if is_source else aarch64_registers
    insn = add_commas(insn, registers)
    return insn

def process_instruction(instruction, is_source=True):
    # 指令按分号分割
    insns = instruction.split(";")
    processed_insns = []
    label_count = 1   
    for insn in insns:
        if 'nop' in insn.lower():
            continue  # 跳过包含 nop 的指令

        processed_insn = process_insn(insn, is_source)
        label_count += processed_insn.count("label")
        processed_insn = processed_insn.replace("label", f"0x{label_count - processed_insn.count('label')}000")
        processed_insns.append(processed_insn)
    
    return ";".join(processed_insns)

# 读取CSV文件
input_file_path = '/home/kingdom/PalmTree-master/angr/test/result_test.csv'
output_file_path = '/home/kingdom/PalmTree-master/angr/test/output.csv'
failed_file_path = '/home/kingdom/PalmTree-master/angr/test/failed.csv'

with open(input_file_path, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile, \
     open(failed_file_path, mode='w', newline='', encoding='utf-8') as failed_file:
    reader = csv.DictReader(infile)
    fieldnames = ['amd64_code', 'aarch64_code']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    failed_writer = csv.DictWriter(failed_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    failed_writer.writeheader()

    success_count = 0
    failed_count = 0

    for row in reader:
        # 预处理并应用规则
        source_preprocessed = preprocess_instruction(row['source'])
        text_preprocessed = preprocess_instruction(row['text'])
        
        amd64_code = process_instruction(source_preprocessed, is_source=True)
        aarch64_code = process_instruction(text_preprocessed, is_source=False)
        ks = Ks(KS_ARCH_X86, KS_MODE_64)
        try:
            encoding, count = ks.asm(amd64_code)
            writer.writerow({'amd64_code': amd64_code, 'aarch64_code': aarch64_code})
            success_count += 1
        except Exception as e:
            # 如果出错，删除 amd64_code 中含有 retn 及其分号的部分，再次尝试
            amd64_code_parts = amd64_code.split(';')
            cleaned_parts = []
            for part in amd64_code_parts:
                if 'retn' not in part.lower():
                    cleaned_parts.append(part)
            cleaned_amd64_code = ';'.join(cleaned_parts)
            try:
                encoding, count = ks.asm(cleaned_amd64_code)
                writer.writerow({'amd64_code': amd64_code, 'aarch64_code': aarch64_code})
                success_count += 1
            except Exception as e:
                failed_writer.writerow({'amd64_code': amd64_code, 'aarch64_code': aarch64_code})
                failed_count += 1

print(f"写入成功数: {success_count}")
print(f"写入失败数: {failed_count}")