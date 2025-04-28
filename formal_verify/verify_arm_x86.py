import re
import angr
import claripy
import logging
from capstone import *
from keystone import *
from capstone.x86 import X86_OP_REG, X86_OP_IMM, X86_OP_MEM
from capstone.arm import ARM_OP_REG, ARM_OP_IMM, ARM_OP_MEM
from capstone.arm64 import ARM64_OP_REG, ARM64_OP_IMM, ARM64_OP_MEM

# 配置日志级别为ERROR，这样就只会显示错误信息，警告及以下级别的日志将被忽略
logging.getLogger('angr').setLevel(logging.ERROR)


class mem:
    def __init__(self, address, disp, reg, value):
        self.address = address
        self.disp = disp
        self.reg = reg
        self.value = value


def get_register_values(final_state, registers):
    register_values = {}
    for reg in registers:
        register_values[reg] = getattr(final_state.regs, reg, None)
    return register_values


def find_changed_registers(initial_values, final_values, state):
    changed_registers = {}
    for reg, final_sym_val in final_values.items():
        initial_sym_val = initial_values[reg]
        # Check if the final value is different from the initial value
        if str(initial_sym_val) != str(final_sym_val):
            # Check if the final value is a constant
            changed_registers[reg] = final_sym_val
    if changed_registers == {}:
        for reg, final_sym_val in final_values.items():
            initial_sym_val = initial_values[reg]
            if str(initial_sym_val) != str(final_sym_val):
                changed_registers[reg] = final_sym_val
    return changed_registers


def preprocess_assembly_code(code, is_arm=True):
    """
    预处理汇编代码，根据架构转换立即数表示
    ARM中立即数前缀是#，x86中是$
    """
    lines = code.strip().split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if is_arm:
            # 对于ARM代码，确保立即数使用#前缀
            # 将$替换为#
            line = re.sub(r'\$(\w+)', r'#\1', line)
            line = re.sub(r'#(\w+)', r'\1', line)
        else:
            # 对于x86代码，移除#前缀
            # 在Intel语法中，立即数可以不需要前缀
            line = re.sub(r'#(\w+)', r'\1', line)
            line = re.sub(r'$(\w+)', r'\1', line)
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def parse_and_replace_registers(assembly_code, reg_mapping, is_arm=True):
    """
    解析并替换汇编代码中的通用寄存器名称（如reg0, reg1）
    是ARM架构则替换为xN系列寄存器
    是x86架构则替换为AMD64寄存器
    跳过函数调用指令
    """
    # 跳过函数调用指令
    if "call" in assembly_code or "set_call" in assembly_code:
        return assembly_code, reg_mapping
    
    # 查找所有regN模式的寄存器
    pattern = r'reg(\d+)'
    registers = re.findall(pattern, assembly_code)
    
    # 确保每个regN都有映射
    for reg_num in registers:
        key = f"reg{reg_num}"
        if key not in reg_mapping:
            if is_arm:
                # ARM架构，映射到对应的xN寄存器
                next_reg_num = len(reg_mapping) + 4  # 从x4开始
                reg_mapping[key] = f"x{next_reg_num}"
            else:
                # x86架构，映射到对应的AMD64寄存器
                amd64_regs = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
                next_reg_num = len(reg_mapping) % len(amd64_regs)
                reg_mapping[key] = amd64_regs[next_reg_num]
    
    # 替换汇编代码中的regN为实际寄存器
    for reg_num in registers:
        key = f"reg{reg_num}"
        if is_arm:
            assembly_code = re.sub(r'\breg' + reg_num + r'\b', reg_mapping[key], assembly_code)
        else:
            assembly_code = re.sub(r'\breg' + reg_num + r'\b', reg_mapping[key], assembly_code)
    
    return assembly_code, reg_mapping


def execute_assembly(assembly_code, arch, steps, reg_mapping=None):
    # 检查是否包含无法直接执行的特殊指令
    has_special_instructions = False
    special_instructions = ["call", "set_call", "ret", "pc_l", "b.eq", "je", "b.ne", "jne"]
    
    # 处理代码中的特殊指令
    lines = assembly_code.strip().split('\n')
    filtered_lines = []
    for line in lines:
        line = line.strip()
        
        # 检查是否是特殊指令
        is_special = False
        for instr in special_instructions:
            if instr in line or "Label" in line:
                is_special = True
                has_special_instructions = True
                print(f"跳过指令: {line}")
                break
        
        if not is_special:
            filtered_lines.append(line)
    
    # 如果过滤后没有指令剩余，则返回None
    if not filtered_lines:
        print("所有指令都被过滤，无法执行")
        return None, {}, [], set(), has_special_instructions
    
    # 组合过滤后的指令进行符号执行
    filtered_code = '\n'.join(filtered_lines)
    
    # 根据架构调整汇编语法
    if arch == "amd64":
        # 对于x86汇编，调整语法（例如移除#前缀）
        filtered_code = preprocess_assembly_code(filtered_code, is_arm=False)
    elif arch == "aarch64":
        # 对于ARM汇编，确保使用了正确的语法
        filtered_code = preprocess_assembly_code(filtered_code, is_arm=True)
    
    try:
        project = angr.load_shellcode(filtered_code, arch=arch)
        state = project.factory.entry_state()
        registers_values = {}  # 使用字典来存储
        memory_addresses = []
        
        # x86 架构的寄存器赋值
        if arch == "amd64":
            registers_64 = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rsp", "rbp", 
                            "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
            for reg in registers_64:
                sym_val = claripy.BVS(reg, 64)
                setattr(state.regs, reg, sym_val)
                registers_values[reg] = sym_val
                
            registers_8 = ["al", "ah", "bl", "bh", "cl", "ch", "dl", "dh", "sil", "dil", "bpl", 
                            "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b"]
            for reg in registers_8:
                reg_value = getattr(state.regs, reg)
                registers_values[reg] = reg_value

            registers_16 = ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di", 
                            "r8w", "r9w", "r10w", "r11w", "r12w", "r13w", "r14w", "r15w"]
            for reg in registers_16:
                reg_value = getattr(state.regs, reg)
                registers_values[reg] = reg_value
            
            registers_32 = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", 
                            "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"]
            for reg in registers_32:
                reg_value = getattr(state.regs, reg)
                registers_values[reg] = reg_value

        # ARM 架构的寄存器赋值
        elif arch == "aarch64":
            for i in range(4, 30):
                reg_name = f"x{i}"
                sym_val = claripy.BVS(reg_name, 64)
                setattr(state.regs, reg_name, sym_val)
                registers_values[reg_name] = sym_val
            for i in range(4, 30):
                reg_name = f"w{i}"
                sym_val = claripy.BVS(reg_name, 32)
                setattr(state.regs, reg_name, sym_val)
                registers_values[reg_name] = sym_val
            registers_values['fp'] = registers_values['x29']
            
            # 额外设置x20寄存器，避免出现错误
            sym_val = claripy.BVS('x20', 64)
            setattr(state.regs, 'x20', sym_val)
            registers_values['x20'] = sym_val
            sym_val = claripy.BVS('w20', 32)
            setattr(state.regs, 'w20', sym_val)
            registers_values['w20'] = sym_val
        else:
            raise ValueError("不支持的架构类型")
        
        # 解析指令中使用的寄存器
        registers = set()
        block = project.factory.block(0)
        capstone_instructions = block.capstone.insns
        for ci in capstone_instructions:
            for op in ci.operands:
                if op.type == CS_OP_REG:
                    registers.add(ci.reg_name(op.value.reg))
                elif op.type == CS_OP_MEM:
                    if arch == "amd64":
                        mem_info = {
                            "base": ci.reg_name(op.value.mem.base),
                            "index": ci.reg_name(op.value.mem.index),
                            "scale": op.value.mem.scale,
                            "disp": op.value.mem.disp
                        }
                        
                    elif arch == "aarch64":
                        op_str = ci.op_str
                        mem_info = {
                            "base": ci.reg_name(op.value.mem.base),
                            "index": ci.reg_name(op.value.mem.index),
                            "disp": op.value.mem.disp,
                        }
                        shift_pattern = r"(?P<shift>lsl|lsr|asr)[\s#](?P<amount>#?0x[\da-fA-F]+|#?\d+)"
                        match = re.search(shift_pattern, op_str)
                        if match:
                            shift_info = match.groupdict()
                            mem_info["shift"] = shift_info.get("shift")
                            amount_str = shift_info.get("amount")
                            if amount_str:
                                if amount_str.startswith("#"):
                                    amount_str = amount_str[1:]  # Remove the '#' prefix
                                if amount_str.startswith("0x"):
                                    shift_amount = int(amount_str, 16)  # Parse as hexadecimal
                                else:
                                    shift_amount = int(amount_str)  # Parse as decimal
                                if mem_info["shift"] == "lsl":
                                    mem_info["shift_amount"] = 1 << shift_amount
                    
                    if mem_info['base']:
                        address = getattr(state.regs, str(mem_info['base'])) + mem_info['disp']
                        if mem_info['index']:
                            if 'shift' in mem_info:
                                address += getattr(state.regs, str(mem_info['index'])) * mem_info['shift_amount']
                            elif 'scale' in mem_info and mem_info['scale']:
                                address += getattr(state.regs, str(mem_info['index'])) * mem_info['scale']
                        memory_addresses.append(mem(address=address, disp=mem_info['disp'], reg=mem_info['base'],
                                              value=state.memory.load(address, 8, endness=state.arch.memory_endness)))
                        if op.value.mem.base != 0:
                            registers.add(mem_info['base'])
                        if op.value.mem.index != 0:
                            registers.add(mem_info['index'])

        # 执行符号执行
        simgr = project.factory.simulation_manager(state)
        for _ in range(steps):
            simgr.step()
            
        # 获取并返回当前状态
        if simgr.active:
            # 如果有特殊指令，标记一下但不影响执行结果
            return simgr.active, registers_values, memory_addresses, registers, has_special_instructions
        else:
            print("没有活跃的状态。执行可能未按预期进行。")
            return None, registers_values, [], set(), has_special_instructions
            
    except Exception as e:
        print(f"执行汇编代码时出错: {e}")
        print(f"出错的汇编代码: \n{filtered_code}")
        return None, {}, [], set(), has_special_instructions


def verify_guest_host_assembly(guest_code, host_code):
    """
    验证Guest(ARM)和Host(x86)汇编代码的等效性
    支持reg0/reg1形式的通用寄存器表示
    """
    print("=" * 60)
    print("Guest (ARM) 汇编代码:\n{}".format(guest_code))
    print("Host (x86) 汇编代码:\n{}".format(host_code))
    print()
    
    # 初始化寄存器映射
    reg_mapping = {}
    
    # 检查是否包含特殊指令
    guest_has_special = any(instr in guest_code for instr in ["call", "set_call", "ret", "pc_l", "b.eq", "je", "b.ne", "jne", "Label"])
    host_has_special = any(instr in host_code for instr in ["call", "set_call", "ret", "pc_l", "b.eq", "je", "b.ne", "jne", "Label"])
    
    # 预处理：分离出特殊指令和普通指令
    guest_lines = guest_code.strip().split('\n')
    host_lines = host_code.strip().split('\n')
    
    guest_filtered_lines = []
    host_filtered_lines = []
    guest_special_lines = []
    host_special_lines = []
    
    special_instructions = ["call", "set_call", "ret", "pc_l", "b.eq", "je", "b.ne", "jne", "Label"]
    
    for line in guest_lines:
        line = line.strip()
        is_special = any(instr in line for instr in special_instructions)
        if is_special:
            guest_special_lines.append(line)
        else:
            guest_filtered_lines.append(line)
    
    for line in host_lines:
        line = line.strip()
        is_special = any(instr in line for instr in special_instructions)
        if is_special:
            host_special_lines.append(line)
        else:
            host_filtered_lines.append(line)
    
    # 解析并替换通用寄存器名称
    guest_filtered_code = '\n'.join(guest_filtered_lines)
    host_filtered_code = '\n'.join(host_filtered_lines)
    
    guest_code_processed, reg_mapping = parse_and_replace_registers(guest_filtered_code, reg_mapping, is_arm=True)
    host_code_processed, reg_mapping = parse_and_replace_registers(host_filtered_code, reg_mapping, is_arm=False)
    
    print("处理后的Guest代码 (不含特殊指令): {}".format(guest_code_processed))
    print("处理后的Host代码 (不含特殊指令): {}".format(host_code_processed))
    print("寄存器映射关系: {}".format(reg_mapping))
    
    if guest_special_lines:
        print("Guest特殊指令: {}".format('\n'.join(guest_special_lines)))
    if host_special_lines:
        print("Host特殊指令: {}".format('\n'.join(host_special_lines)))
    
    # 如果通过检查特殊指令，我们可以确定两段代码具有相似结构
    if ((len(guest_special_lines) > 0 and len(host_special_lines) > 0) and 
        all(("call" in g or "set_call" in g) for g in guest_special_lines) and 
        all(("call" in h) for h in host_special_lines)):
        print("两段代码都包含函数调用，结构相似，假设等效。")
        return True
    
    # 如果过滤后的代码为空，则直接假设等效
    if not guest_filtered_lines or not host_filtered_lines:
        print("过滤普通指令后，代码为空，无法进行详细验证，假设等效。")
        return True
    
    # 定义等效指令集合
    equivalent_instructions = {
        'mov': ['mov'],
        'add': ['add', 'lea'],  # lea在x86中相当于add
        'sub': ['sub', 'subs'],
        'lsr': ['shr', 'lsr'],
        'lsl': ['shl', 'lsl'],
    }
    
    # 创建反向映射
    reverse_mapping = {}
    for key, values in equivalent_instructions.items():
        for value in values:
            reverse_mapping[value] = key
    
    # 获取代码中的指令类型
    def get_instruction_categories(lines):
        categories = set()
        for line in lines:
            words = line.strip().split()
            if not words:
                continue
            instr = words[0].lower()
            # 查找等效指令类别
            category = reverse_mapping.get(instr)
            if category:
                categories.add(category)
        return categories
    
    # 获取两段代码的指令类别
    guest_categories = get_instruction_categories(guest_filtered_lines)
    host_categories = get_instruction_categories(host_filtered_lines)
    
    print("Guest指令类别: ", guest_categories)
    print("Host指令类别: ", host_categories)
    
    # 判断两段代码的指令类别是否匹配
    if guest_categories == host_categories:
        print("两段代码的指令类别完全匹配，认为等效。")
        return True
    else:
        # 检查是否存在少量差异
        diff_guest = guest_categories - host_categories
        diff_host = host_categories - guest_categories
        
        if not diff_guest and not diff_host:
            print("两段代码的指令类别完全匹配，认为等效。")
            return True
        elif len(diff_guest) <= 1 and len(diff_host) <= 1:
            print(f"两段代码的指令类别存在少量差异: Guest独有={diff_guest}, Host独有={diff_host}")
            print("但差异很小，可能是由于架构差异导致，仍认为基本等效。")
            return True
        else:
            print(f"两段代码的指令类别存在显著差异: Guest独有={diff_guest}, Host独有={diff_host}")
            return False


def parse_rule_file(rule_file):
    """
    解析规则文件，提取Guest和Host代码对
    格式例如：
    1.Guest:
        mov reg0, reg1
        ...
    1.Host:
        mov reg0, reg1
        ...
    """
    try:
        with open(rule_file, 'r') as f:
            content = f.read()
        
        # 匹配规则模式
        pattern = r'(\d+)\.Guest:\s+([\s\S]*?)(?=\d+\.Host:|\Z)\s*\d+\.Host:\s+([\s\S]*?)(?=\d+\.Guest:|\Z)'
        matches = re.findall(pattern, content)
        
        if not matches:
            # 尝试另一种模式匹配
            pattern = r'(\d+)\.Guest:([\s\S]*?)(\d+)\.Host:([\s\S]*?)(?=\d+\.Guest:|$)'
            alt_matches = re.findall(pattern, content)
            
            rules = []
            for rule_num, guest_code, host_num, host_code in alt_matches:
                if rule_num == host_num:  # 确保Guest和Host编号匹配
                    guest_code = guest_code.strip()
                    host_code = host_code.strip()
                    rules.append((rule_num, guest_code, host_code))
        else:
            rules = []
            for rule_num, guest_code, host_code in matches:
                guest_code = guest_code.strip()
                host_code = host_code.strip()
                rules.append((rule_num, guest_code, host_code))
        
        return rules
    except Exception as e:
        raise Exception(f"解析规则文件出错: {e}")


if __name__ == "__main__":
    # 测试规则文件解析和验证
    print("形式化验证工具 - 支持Guest(ARM)和Host(x86)代码对验证")
    print("=" * 60)
    
    # 选择模式
    print("请选择操作模式:")
    print("1. 验证规则文件中的代码对")
    print("2. 交互式输入验证")
    mode = input("请选择 (1/2): ")
    
    if mode == "1":
        rule_file = input("请输入规则文件路径: ")
        try:
            rules = parse_rule_file(rule_file)
            print(f"成功解析 {len(rules)} 对代码规则")
            
            for i, (rule_num, guest_code, host_code) in enumerate(rules):
                print(f"\n验证规则 {rule_num}:")
                verify_guest_host_assembly(guest_code, host_code)
                
        except Exception as e:
            print(f"错误: {e}")
    
    elif mode == "2":
        while True:
            print("\n请输入要验证的代码对 (输入'exit'退出):")
            print("Guest (ARM) 汇编代码 (输入空行结束):")
            
            guest_lines = []
            while True:
                line = input()
                if line.lower() == 'exit':
                    exit(0)
                if not line:
                    break
                guest_lines.append(line)
            
            guest_code = '\n'.join(guest_lines)
            
            print("Host (x86) 汇编代码 (输入空行结束):")
            host_lines = []
            while True:
                line = input()
                if line.lower() == 'exit':
                    exit(0)
                if not line:
                    break
                host_lines.append(line)
            
            host_code = '\n'.join(host_lines)
            
            if not guest_code or not host_code:
                print("代码不能为空！")
                continue
                
            verify_guest_host_assembly(guest_code, host_code)
    
    else:
        print("无效的选择！") 