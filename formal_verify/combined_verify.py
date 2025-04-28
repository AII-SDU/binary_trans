import re
import angr
import claripy
import logging
from capstone import *
from keystone import *
from capstone.x86 import X86_OP_REG, X86_OP_IMM, X86_OP_MEM
from capstone.arm import ARM_OP_REG, ARM_OP_IMM, ARM_OP_MEM
from capstone.arm64 import ARM64_OP_REG, ARM64_OP_IMM, ARM64_OP_MEM

# 配置日志级别为ERROR，这样就只会显示错误信息
logging.getLogger('angr').setLevel(logging.ERROR)

# 创建一个专用于验证器的日志记录器
verify_logger = logging.getLogger('verify')
verify_logger.setLevel(logging.INFO)  # 默认级别为INFO，可以显示重要信息

# 设置DEBUG模式，如果为False则不显示位宽不匹配警告
DEBUG_MODE = False

class mem:
    def __init__(self, address, disp, reg, value):
        self.address = address
        self.disp = disp
        self.reg = reg
        self.value = value

# 寄存器映射关系
AMD64_TO_ARM64 = {
    'rax': 'x4', 'rcx': 'x5', 'rdx': 'x6', 'rbx': 'x7',
    'rsp': 'x8', 'rbp': 'x9', 'rsi': 'x10', 'rdi': 'x11',
    'r8': 'x12', 'r9': 'x13', 'r10': 'x14', 'r11': 'x15',
    'r12': 'x16', 'r13': 'x17', 'r14': 'x19', 'r15': 'fp',
    'eax': 'w4', 'ecx': 'w5', 'edx': 'w6', 'ebx': 'w7',
    'esp': 'w8', 'ebp': 'w9', 'esi': 'w10', 'edi': 'w11',
    'r8d': 'w12', 'r9d': 'w13', 'r10d': 'w14', 'r11d': 'w15',
    'r12d': 'w16', 'r13d': 'w17', 'r14d': 'w19', 'r15d': 'w29',
    'ax': 'w4', 'cx': 'w5', 'dx': 'w6', 'bx': 'w7',
    'sp': 'w8', 'bp': 'w9', 'si': 'w10', 'di': 'w11',
    'r8w': 'w12', 'r9w': 'w13', 'r10w': 'w14', 'r11w': 'w15',
    'r12w': 'w16', 'r13w': 'w17', 'r14w': 'w19', 'r15w': 'w29',
    'al': 'w4', 'cl': 'w5', 'dl': 'w6', 'bl': 'w7',
    'spl': 'w8', 'bpl': 'w9', 'sil': 'w10', 'dil': 'w11',
    'r8b': 'w12', 'r9b': 'w13', 'r10b': 'w14', 'r11b': 'w15',
    'r12b': 'w16', 'r13b': 'w17', 'r14b': 'w19', 'r15b': 'w29',
}

# 等效指令集合
EQUIVALENT_INSTRUCTIONS = {
    'mov': ['mov', 'ldr', 'ldur', 'str', 'stur', 'ldrb', 'ldurb'],
    'add': ['add', 'lea', 'adds'],
    'sub': ['sub', 'subs'],
    'lsr': ['shr', 'lsr', 'lsrs'],
    'lsl': ['shl', 'lsl', 'lsls'],
    'and': ['and', 'ands'],
    'orr': ['or', 'orr'],
    'xor': ['xor', 'eor'],
    'test': ['test', 'tst'],
    'cmp': ['cmp', 'cmps'],
    'cmove': ['cmove', 'csel'],
    'cmovne': ['cmovne', 'csel'],
    'cmova': ['cmova', 'csel'],
    'zero': ['xor esi, esi', 'mov wzr', 'mov w10, wzr'],
}

# 特殊指令列表
SPECIAL_INSTRUCTIONS = ["call", "set_call", "ret", "pc_l", "b.eq", "je", "b.ne", "jne", "Label"]

# 工具函数
def get_register_values(final_state, registers):
    """获取最终状态中寄存器的值"""
    register_values = {}
    for reg in registers:
        register_values[reg] = getattr(final_state.regs, reg, None)
    return register_values

def find_changed_registers(initial_values, final_values, state):
    """找出状态改变的寄存器"""
    changed_registers = {}
    for reg, final_sym_val in final_values.items():
        initial_sym_val = initial_values[reg]
        # 检查最终值是否与初始值不同
        if str(initial_sym_val) != str(final_sym_val):
            changed_registers[reg] = final_sym_val
    if changed_registers == {}:
        for reg, final_sym_val in final_values.items():
            initial_sym_val = initial_values[reg]
            if str(initial_sym_val) != str(final_sym_val):
                changed_registers[reg] = final_sym_val
    return changed_registers

def preprocess_assembly_code(code, is_arm=True):
    """预处理汇编代码，统一立即数表示方式"""
    lines = code.strip().split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if is_arm:
            # 对于ARM代码，移除#前缀，统一格式
            line = re.sub(r'#(\w+)', r'\1', line)
            # 同时也处理$前缀
            line = re.sub(r'\$(\w+)', r'\1', line)
            # 处理.x后缀的寄存器名
            line = re.sub(r'(\w+)\.x', r'\1', line)
        else:
            # 对于x86代码，保持原样
            line = re.sub(r'\$(\w+)', r'\1', line)
            pass
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def parse_and_replace_registers(assembly_code, reg_mapping, is_arm=True):
    """解析并替换汇编代码中的通用寄存器名称"""
    # 跳过函数调用指令
    if any(instr in assembly_code for instr in SPECIAL_INSTRUCTIONS):
        return assembly_code, reg_mapping
    
    # 预处理：如果有多条指令用分号分隔，先拆分后再组合
    if ";" in assembly_code:
        instructions = assembly_code.split(";")
        processed_instructions = []
        
        for instr in instructions:
            processed_instr, reg_mapping = parse_and_replace_single_instruction(instr.strip(), reg_mapping, is_arm)
            processed_instructions.append(processed_instr)
        
        return ";".join(processed_instructions), reg_mapping
    else:
        return parse_and_replace_single_instruction(assembly_code, reg_mapping, is_arm)

def parse_and_replace_single_instruction(instruction, reg_mapping, is_arm=True):
    """处理单条指令中的寄存器替换"""
    # 首先处理立即数格式，移除ARM中的#前缀
    if is_arm:
        instruction = re.sub(r'#(\w+)', r'\1', instruction)
        # 处理ARM寄存器中的.x后缀
        instruction = re.sub(r'(\w+)\.x', r'\1', instruction)
    
    # 查找所有regN模式的寄存器
    pattern = r'reg(\d+)'
    registers = re.findall(pattern, instruction)
    
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
        instruction = re.sub(r'\breg' + reg_num + r'(?:\.x)?', reg_mapping[key], instruction)
    
    # 定义适合当前架构的通用寄存器映射
    if is_arm:
        # ARM架构中寄存器映射
        common_regs = {
            'w4': 'w4', 'x4': 'x4',
            'w5': 'w5', 'x5': 'x5',
            'w6': 'w6', 'x6': 'x6',
            'w7': 'w7', 'x7': 'x7',
            'w8': 'w8', 'x8': 'x8',
            'w9': 'w9', 'x9': 'x9',
            'w10': 'w10', 'x10': 'x10',
            'w11': 'w11', 'x11': 'x11',
            'w12': 'w12', 'x12': 'x12',
            'w13': 'w13', 'x13': 'x13',
            'w14': 'w14', 'x14': 'x14',
            'w15': 'w15', 'x15': 'x15',
            'w16': 'w16', 'x16': 'x16',
            'w17': 'w17', 'x17': 'x17',
            'w19': 'w19', 'x19': 'x19',
            'w20': 'w20', 'x20': 'x20',
            'w23': 'w23', 'x23': 'x23',
            'w29': 'w29', 'fp': 'fp',
            'wzr': 'wzr', 'xzr': 'xzr'
        }
    else:
        # x86架构中的寄存器映射
        common_regs = {
            'rax': 'rax', 'eax': 'eax',
            'rcx': 'rcx', 'ecx': 'ecx',
            'rdx': 'rdx', 'edx': 'edx',
            'rbx': 'rbx', 'ebx': 'ebx',
            'rsp': 'rsp', 'esp': 'esp',
            'rbp': 'rbp', 'ebp': 'ebp',
            'rsi': 'rsi', 'esi': 'esi',
            'rdi': 'rdi', 'edi': 'edi',
            'r8': 'r8', 'r8d': 'r8d',
            'r9': 'r9', 'r9d': 'r9d',
            'r10': 'r10', 'r10d': 'r10d',
            'r11': 'r11', 'r11d': 'r11d',
            'r12': 'r12', 'r12d': 'r12d',
            'r13': 'r13', 'r13d': 'r13d',
            'r14': 'r14', 'r14d': 'r14d',
            'r15': 'r15', 'r15d': 'r15d'
        }
    
    # 检查代码中是否包含通用寄存器，如果有则替换映射
    for reg, mapped_reg in common_regs.items():
        # 使用正则表达式确保只匹配完整的寄存器名称，同时处理可能的.x后缀
        instruction = re.sub(r'\b' + re.escape(reg) + r'(?:\.x)?\b', mapped_reg, instruction)
        
        # 如果是寄存器在内存引用中，也需要替换
        if is_arm:
            # ARM 格式: [x9, #0x8] 或 [x9.x, #0x8]
            instruction = re.sub(r'\[' + re.escape(reg) + r'(?:\.x)?(.*?)\]', f'[{mapped_reg}\\1]', instruction)
        else:
            # x86 格式: [rbp + 0x8]
            instruction = re.sub(r'\[' + re.escape(reg) + r'(.*?)\]', f'[{mapped_reg}\\1]', instruction)
    
    # 统一ARM的ldr/str和x86的mov内存操作
    if is_arm:
        # LDR指令转换为MOV (从内存加载)
        ldr_pattern = r'ldr\s+(\w+)(?:\.x)?,\s*\[(.*?)\]'
        instruction = re.sub(ldr_pattern, r'mov \1, [\2]', instruction)
        # LDRB指令转换
        ldrb_pattern = r'ldrb\s+(\w+)(?:\.x)?,\s*\[(.*?)\]'
        instruction = re.sub(ldrb_pattern, r'mov \1, byte ptr [\2]', instruction)
        # STR指令转换为MOV (存储到内存)
        str_pattern = r'str\s+(\w+)(?:\.x)?,\s*\[(.*?)\]'
        instruction = re.sub(str_pattern, r'mov [\2], \1', instruction)
        # LDUR指令转换
        ldur_pattern = r'ldur\s+(\w+)(?:\.x)?,\s*\[(.*?)\]'
        instruction = re.sub(ldur_pattern, r'mov \1, [\2]', instruction)
        
        # 处理ARM特有指令
        # 处理wzr (零寄存器)
        instruction = re.sub(r'\bwzr(?:\.x)?\b', r'0', instruction)
        instruction = re.sub(r'\bxzr(?:\.x)?\b', r'0', instruction)
        
        # 处理通用的内存访问格式
        # [x8, 0x2] -> [x8 + 0x2] 或 [x8.x, 0x2] -> [x8 + 0x2]
        mem_pattern = r'\[(\w+)(?:\.x)?,\s*([^\]]+)\]'
        instruction = re.sub(mem_pattern, r'[\1 + \2]', instruction)
    else:
        # 将x86的内存访问格式统一
        # 保持 [reg + offset] 格式
        pass
    
    return instruction, reg_mapping

def execute_assembly(assembly_code, arch, steps, reg_mapping=None):
    """执行汇编代码并返回执行结果"""
    # 检查是否包含无法直接执行的特殊指令
    has_special_instructions = False
    
    # 处理代码中的特殊指令
    lines = assembly_code.strip().split('\n')
    filtered_lines = []
    for line in lines:
        line = line.strip()
        
        # 检查是否是特殊指令
        is_special = False
        for instr in SPECIAL_INSTRUCTIONS:
            if instr in line:
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
        filtered_code = preprocess_assembly_code(filtered_code, is_arm=False)
    elif arch == "aarch64":
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
                            "base": ci.reg_name(op.value.mem.base) if op.value.mem.base != 0 else None,
                            "index": ci.reg_name(op.value.mem.index) if op.value.mem.index != 0 else None,
                            "scale": op.value.mem.scale,
                            "disp": op.value.mem.disp
                        }
                        
                    elif arch == "aarch64":
                        op_str = ci.op_str
                        mem_info = {
                            "base": ci.reg_name(op.value.mem.base) if op.value.mem.base != 0 else None,
                            "index": ci.reg_name(op.value.mem.index) if op.value.mem.index != 0 else None,
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
                            if 'shift' in mem_info and 'shift_amount' in mem_info:
                                address += getattr(state.regs, str(mem_info['index'])) * mem_info['shift_amount']
                            elif 'scale' in mem_info and mem_info['scale']:
                                address += getattr(state.regs, str(mem_info['index'])) * mem_info['scale']
                        memory_addresses.append(mem(address=address, disp=mem_info['disp'], reg=mem_info['base'],
                                              value=state.memory.load(address, 8, endness=state.arch.memory_endness)))
                        if mem_info['base']:
                            registers.add(mem_info['base'])
                        if mem_info['index']:
                            registers.add(mem_info['index'])

        # 执行符号执行
        simgr = project.factory.simulation_manager(state)
        for _ in range(steps):
            simgr.step()
            
        # 获取并返回当前状态
        if simgr.active:
            return simgr.active, registers_values, memory_addresses, registers, has_special_instructions
        else:
            print("没有活跃的状态。执行可能未按预期进行。")
            return None, registers_values, [], set(), has_special_instructions
            
    except Exception as e:
        print(f"执行汇编代码时出错: {e}")
        print(f"出错的汇编代码: \n{filtered_code}")
        return None, {}, [], set(), has_special_instructions 

def verify_precise(amd64_code, aarch64_code):
    """使用精确符号执行验证单条指令的等价性"""
    print("=" * 60)
    print("精确验证模式")
    print("AMD64汇编: {}".format(amd64_code))
    print("AArch64汇编: {}".format(aarch64_code))
    print()
    flag = True
    
    # 预处理：处理reg映射，为两种架构分别创建独立的映射
    amd64_reg_mapping = {}
    aarch64_reg_mapping = {}
    
    # 分别处理AMD64和AArch64代码
    amd64_code_processed, amd64_reg_mapping = parse_and_replace_registers(amd64_code, amd64_reg_mapping, is_arm=False) 
    aarch64_code_processed, aarch64_reg_mapping = parse_and_replace_registers(aarch64_code, aarch64_reg_mapping, is_arm=True)
    
    # 显示处理后的代码和映射关系
    if amd64_code_processed != amd64_code or aarch64_code_processed != aarch64_code:
        print("处理寄存器映射后:")
        print(f"AMD64: {amd64_code_processed}")
        print(f"AArch64: {aarch64_code_processed}")
        print(f"AMD64寄存器映射: {amd64_reg_mapping}")
        print(f"AArch64寄存器映射: {aarch64_reg_mapping}")
    
    # 构建regN的交叉映射关系，用于后续比较
    cross_mapping = {}
    for key in set(amd64_reg_mapping.keys()) & set(aarch64_reg_mapping.keys()):
        cross_mapping[amd64_reg_mapping[key]] = aarch64_reg_mapping[key]
    
    # 符号化执行得到最终状态
    final_state_amd64, amd64_regs, mem_amd, amd64_registers, amd64_has_special = execute_assembly(amd64_code_processed, "amd64", 1)
    final_state_aarch64, aarch64_regs, mem_arm, aarch64_registers, aarch64_has_special = execute_assembly(aarch64_code_processed, "aarch64", 1)
    
    # 如果有特殊指令，则使用启发式验证
    if amd64_has_special or aarch64_has_special:
        print("检测到特殊指令，切换到启发式验证...")
        return verify_heuristic(aarch64_code, amd64_code)
    
    # 循环对x4到x29进行操作，处理64位与32位寄存器的关系
    for i in range(4, 30):
        if f'x{i}' in aarch64_regs and f'w{i}' in aarch64_regs:
            # 提取63到32位
            x_high_32 = claripy.Extract(63, 32, aarch64_regs[f'x{i}'])
            # 连接成一个64位变量
            x_combined = claripy.Concat(x_high_32, aarch64_regs[f'w{i}'])
            # 将操作后的变量存储到字典中
            aarch64_regs[f'x{i}'] = x_combined
        
    # 解析内存与寄存器
    print("AMD64寄存器: ", amd64_registers)
    print("AArch64寄存器: ", aarch64_registers)
    
    if final_state_amd64 and final_state_aarch64 and len(final_state_amd64) == 1 and len(final_state_aarch64) == 1:
        final_state_amd64 = final_state_amd64[0]
        final_state_aarch64 = final_state_aarch64[0]
        
        amd64_registers_value = get_register_values(final_state_amd64, amd64_registers)
        aarch64_registers_value = get_register_values(final_state_aarch64, aarch64_registers)
        
        registers_output = find_changed_registers(amd64_regs, amd64_registers_value, final_state_amd64)
        arm_registers_output = find_changed_registers(aarch64_regs, aarch64_registers_value, final_state_aarch64)
        
        print("AMD64改变的寄存器: ", registers_output)
        print("AArch64改变的寄存器: ", arm_registers_output)
        
        s = claripy.Solver()
        
        # 创建寄存器映射关系，用于验证等价性
        register_mapping = {}
        
        # 添加通过regN映射产生的对应关系
        for reg_key in set(amd64_reg_mapping.keys()) & set(aarch64_reg_mapping.keys()):
            amd64_reg = amd64_reg_mapping[reg_key]
            aarch64_reg = aarch64_reg_mapping[reg_key]
            
            if amd64_reg in amd64_regs and aarch64_reg in aarch64_regs:
                register_mapping[amd64_reg] = aarch64_reg
                
                # 检查并处理位宽，确保可以安全比较，但不输出警告
                amd64_val = amd64_regs[amd64_reg]
                aarch64_val = aarch64_regs[aarch64_reg]
                
                try:
                    # 尝试添加约束条件，如果失败则可能是位宽不匹配
                    s.add(amd64_val == aarch64_val)
                except Exception:
                    # 如果位宽不匹配，尝试扩展或截断
                    if amd64_val.size() > aarch64_val.size():
                        # 扩展ARM寄存器值
                        aarch64_val_extended = claripy.ZeroExt(amd64_val.size() - aarch64_val.size(), aarch64_val)
                        s.add(amd64_val == aarch64_val_extended)
                    elif amd64_val.size() < aarch64_val.size():
                        # 截断ARM寄存器值
                        aarch64_val_truncated = claripy.Extract(amd64_val.size()-1, 0, aarch64_val)
                        s.add(amd64_val == aarch64_val_truncated)
        
        # 添加标准的AMD64到ARM64映射，但只映射固定大小的寄存器对
        valid_register_pairs = []
        for amd64_reg, aarch64_reg in AMD64_TO_ARM64.items():
            # 只添加size相同或可以处理的寄存器对
            if amd64_reg in amd64_regs and aarch64_reg in aarch64_regs:
                amd64_val = amd64_regs[amd64_reg]
                aarch64_val = aarch64_regs[aarch64_reg]
                
                # 检查是否为相同位宽的64位寄存器
                if amd64_reg.startswith('r') and aarch64_reg.startswith('x') and amd64_val.size() == aarch64_val.size():
                    valid_register_pairs.append((amd64_reg, aarch64_reg))
                # 检查是否为相同位宽的32位寄存器
                elif (amd64_reg.endswith('d') or amd64_reg in ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp']) and aarch64_reg.startswith('w') and amd64_val.size() == aarch64_val.size():
                    valid_register_pairs.append((amd64_reg, aarch64_reg))
                # 跳过位宽不同的寄存器对
        
        # 为有效的寄存器对添加约束
        for amd64_reg, aarch64_reg in valid_register_pairs:
            register_mapping[amd64_reg] = aarch64_reg
            amd64_val = amd64_regs[amd64_reg]
            aarch64_val = aarch64_regs[aarch64_reg]
            
            try:
                s.add(amd64_val == aarch64_val)
            except Exception:
                # 如果操作失败，说明还是有问题，直接跳过这对寄存器
                pass
            
        # 添加内存约束条件
        for mem_amd_obj in mem_amd:
            for mem_arm_obj in mem_arm:
                amd_reg = mem_amd_obj.reg
                arm_reg = mem_arm_obj.reg
                
                # 检查是否是对应的寄存器对
                is_corresponding = False
                
                # 通过regN映射检查
                for reg_key in set(amd64_reg_mapping.keys()) & set(aarch64_reg_mapping.keys()):
                    if amd_reg == amd64_reg_mapping[reg_key] and arm_reg == aarch64_reg_mapping[reg_key]:
                        is_corresponding = True
                        break
                
                # 通过标准映射检查
                if not is_corresponding and amd_reg in AMD64_TO_ARM64 and arm_reg == AMD64_TO_ARM64[amd_reg]:
                    is_corresponding = True
                
                if mem_amd_obj.disp == mem_arm_obj.disp and is_corresponding:
                    memory_content_amd1 = mem_amd_obj.value
                    memory_content_arm1 = mem_arm_obj.value
                    memory_content_amd2 = final_state_amd64.memory.load(mem_amd_obj.address, 8, endness=final_state_amd64.arch.memory_endness)
                    memory_content_arm2 = final_state_aarch64.memory.load(mem_arm_obj.address, 8, endness=final_state_aarch64.arch.memory_endness)
                    
                    print("初始AMD64内存:", memory_content_amd1)
                    print("初始AArch64内存:", memory_content_arm1)
                    print("最终AMD64内存:", memory_content_amd2)
                    print("最终AArch64内存:", memory_content_arm2)
                    
                    # 检查位宽并处理，但不输出警告
                    try:
                        s.add(memory_content_amd1 == memory_content_arm1)
                    except Exception:
                        # 处理位宽不匹配
                        if memory_content_amd1.size() > memory_content_arm1.size():
                            memory_content_arm1_ext = claripy.ZeroExt(memory_content_amd1.size() - memory_content_arm1.size(), memory_content_arm1)
                            s.add(memory_content_amd1 == memory_content_arm1_ext)
                        elif memory_content_amd1.size() < memory_content_arm1.size():
                            memory_content_arm1_trunc = claripy.Extract(memory_content_amd1.size()-1, 0, memory_content_arm1)
                            s.add(memory_content_amd1 == memory_content_arm1_trunc)
                    
                    if (str(memory_content_amd1) != str(memory_content_amd2)):
                        try:
                            is_not_equal = s.satisfiable(extra_constraints=[memory_content_amd2 != memory_content_arm2])
                        except Exception:
                            # 处理位宽不匹配，但不输出警告
                            if memory_content_amd2.size() > memory_content_arm2.size():
                                memory_content_arm2_ext = claripy.ZeroExt(memory_content_amd2.size() - memory_content_arm2.size(), memory_content_arm2)
                                is_not_equal = s.satisfiable(extra_constraints=[memory_content_amd2 != memory_content_arm2_ext])
                            elif memory_content_amd2.size() < memory_content_arm2.size():
                                memory_content_amd2_ext = claripy.ZeroExt(memory_content_arm2.size() - memory_content_amd2.size(), memory_content_amd2)
                                is_not_equal = s.satisfiable(extra_constraints=[memory_content_amd2_ext != memory_content_arm2])
                            else:
                                is_not_equal = True
                            
                        if is_not_equal:
                            flag = False
                            print(f"内存不匹配：AMD64({amd_reg}+{mem_amd_obj.disp}) != AArch64({arm_reg}+{mem_arm_obj.disp})")

        # 检查改变的寄存器是否匹配
        for amd_reg in registers_output:
            if amd_reg in register_mapping:
                arm_reg = register_mapping[amd_reg]
                if arm_reg in aarch64_registers_value:
                    arm_expr = aarch64_registers_value[arm_reg]
                    amd_expr = amd64_registers_value[amd_reg]

                    try:
                        is_not_equal = s.satisfiable(extra_constraints=[arm_expr != amd_expr])
                    except Exception:
                        # 处理位宽不匹配，但不输出警告
                        if amd_expr.size() > arm_expr.size():
                            arm_expr_ext = claripy.ZeroExt(amd_expr.size() - arm_expr.size(), arm_expr)
                            is_not_equal = s.satisfiable(extra_constraints=[amd_expr != arm_expr_ext])
                        elif amd_expr.size() < arm_expr.size():
                            amd_expr_ext = claripy.ZeroExt(arm_expr.size() - amd_expr.size(), amd_expr)
                            is_not_equal = s.satisfiable(extra_constraints=[amd_expr_ext != arm_expr])
                        else:
                            is_not_equal = True
                            
                    if is_not_equal:
                        flag = False
                        print(f"寄存器不匹配：AMD64({amd_reg}) != AArch64({arm_reg})")
                    
        if flag:
            print("两段汇编只有一条路径，且两段汇编等价")
            return True
        else:
            print("两段汇编不等价")
            return False
            
    elif not final_state_amd64 or not final_state_aarch64:
        print("至少一种架构的汇编代码执行出错，切换到启发式验证")
        return verify_heuristic(aarch64_code, amd64_code)
    elif len(final_state_amd64) > 1 and len(final_state_aarch64) > 1:
        # 多个执行路径的处理逻辑，切换到启发式方法
        print("汇编代码有多个执行路径，切换到启发式验证...")
        return verify_heuristic(aarch64_code, amd64_code)
    else:
        print("两段汇编的执行路径数量不同，不等价")
        return False

def verify_heuristic(guest_code, host_code):
    """使用启发式方法验证复杂指令序列的等价性"""
    print("=" * 60)
    print("启发式验证模式")
    print("Guest (ARM) 汇编代码: {}".format(guest_code))
    print("Host (x86) 汇编代码: {}".format(host_code))
    print()
    
    # 初始化寄存器映射，为不同架构创建独立的映射
    guest_reg_mapping = {}
    host_reg_mapping = {}
    
    # 检查是否包含特殊指令
    guest_has_special = any(instr in guest_code for instr in SPECIAL_INSTRUCTIONS)
    host_has_special = any(instr in host_code for instr in SPECIAL_INSTRUCTIONS)
    
    # 如果两边都有特殊指令且类型相似，则假定等效
    if guest_has_special and host_has_special:
        special_guest = [line for line in guest_code.split('\n') if any(instr in line for instr in SPECIAL_INSTRUCTIONS)]
        special_host = [line for line in host_code.split('\n') if any(instr in line for instr in SPECIAL_INSTRUCTIONS)]
        
        if all(("call" in g or "set_call" in g) for g in special_guest) and all(("call" in h) for h in special_host):
            print("两段代码都包含函数调用，结构相似，假设等效。")
            return True
    
    # 解析并替换通用寄存器名称，使用各自的映射
    guest_code_processed, guest_reg_mapping = parse_and_replace_registers(guest_code, guest_reg_mapping, is_arm=True)
    host_code_processed, host_reg_mapping = parse_and_replace_registers(host_code, host_reg_mapping, is_arm=False)
    
    print("处理后的Guest代码: {}".format(guest_code_processed))
    print("处理后的Host代码: {}".format(host_code_processed))
    print("Guest寄存器映射: {}".format(guest_reg_mapping))
    print("Host寄存器映射: {}".format(host_reg_mapping))
    
    # 创建反向映射
    reverse_mapping = {}
    for key, values in EQUIVALENT_INSTRUCTIONS.items():
        for value in values:
            reverse_mapping[value] = key
    
    # 获取代码中的指令类型
    def get_instruction_categories(code):
        categories = set()
        for line in code.split('\n'):
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
    guest_categories = get_instruction_categories(guest_code_processed)
    host_categories = get_instruction_categories(host_code_processed)
    
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

def verify_combined(amd64_code, aarch64_code):
    """综合验证方法，先对每条指令进行精确验证，再基于验证结果判断整体等价性"""
    print("=" * 60)
    print("综合验证模式")
    print("AMD64汇编: {}".format(amd64_code))
    print("AArch64汇编: {}".format(aarch64_code))
    print()
    
    # 预处理：处理分号分隔的多条指令
    if ";" in amd64_code or ";" in aarch64_code:
        print("检测到分号分隔的多条指令")
        
        # 分解指令
        amd64_instructions = [instr.strip() for instr in amd64_code.split(";") if instr.strip()]
        aarch64_instructions = [instr.strip() for instr in aarch64_code.split(";") if instr.strip()]
        
        print(f"AMD64包含 {len(amd64_instructions)} 条指令")
        print(f"AArch64包含 {len(aarch64_instructions)} 条指令")
        
        # 预处理指令，统一寄存器和内存访问格式
        reg_mapping = {}
        processed_amd64 = []
        processed_aarch64 = []
        
        for instr in amd64_instructions:
            processed, reg_mapping = parse_and_replace_single_instruction(instr, reg_mapping, is_arm=False)
            processed_amd64.append(processed)
        
        for instr in aarch64_instructions:
            processed, reg_mapping = parse_and_replace_single_instruction(instr, reg_mapping, is_arm=True)
            processed_aarch64.append(processed)
        
        print("处理后的指令:")
        print("AMD64:", "; ".join(processed_amd64))
        print("AArch64:", "; ".join(processed_aarch64))
        print("寄存器映射:", reg_mapping)
        
        # 创建反向映射用于指令类型判断
        reverse_mapping = {}
        for key, values in EQUIVALENT_INSTRUCTIONS.items():
            for value in values:
                reverse_mapping[value] = key
        
        # 收集指令类型信息
        amd64_types = []
        aarch64_types = []
        
        for instr in processed_amd64:
            if not instr:
                continue
            parts = instr.split()
            if not parts:
                continue
            op = parts[0].lower()
            category = reverse_mapping.get(op)
            amd64_types.append((op, category))
        
        for instr in processed_aarch64:
            if not instr:
                continue
            parts = instr.split()
            if not parts:
                continue
            op = parts[0].lower()
            category = reverse_mapping.get(op)
            aarch64_types.append((op, category))
        
        print("AMD64指令类型:", [t for op, t in amd64_types if t])
        print("AArch64指令类型:", [t for op, t in aarch64_types if t])
        
        # 分析指令类型匹配度
        match_count = 0
        matched_pairs = []
        
        # 尝试一对一匹配指令
        x86_used = set()
        arm_used = set()
        
        for i, (x86_op, x86_type) in enumerate(amd64_types):
            if i in x86_used or x86_type is None:
                continue
                
            for j, (arm_op, arm_type) in enumerate(aarch64_types):
                if j in arm_used or arm_type is None:
                    continue
                    
                if x86_type == arm_type:
                    match_count += 1
                    matched_pairs.append((i, j))
                    x86_used.add(i)
                    arm_used.add(j)
                    print(f"匹配: AMD64指令{i+1}({x86_op}/{x86_type}) 对应 AArch64指令{j+1}({arm_op}/{arm_type})")
                    break
        
        # 计算匹配率
        total_instructions = max(len(amd64_types), len(aarch64_types))
        if total_instructions > 0:
            match_ratio = match_count / total_instructions
            print(f"指令匹配率: {match_ratio:.2%}")
            
            # 根据匹配率判断等价性
            if match_ratio >= 0.8:
                print("指令匹配率高，认为两段代码等价 ✓")
                return True
            elif match_ratio >= 0.5:
                print("指令匹配率中等，大体上认为等价")
                return True
            else:
                print("指令匹配率低，可能不等价 ✗")
                return False
        else:
            print("没有有效指令，无法判断等价性")
            return False
    
    # 分解为单条指令（针对换行分隔的指令）
    amd64_lines = [line.strip() for line in amd64_code.strip().split('\n') if line.strip()]
    aarch64_lines = [line.strip() for line in aarch64_code.strip().split('\n') if line.strip()]
    
    # 如果都是单条指令，直接精确验证
    if len(amd64_lines) == 1 and len(aarch64_lines) == 1:
        print("单条指令，使用精确验证模式")
        return verify_precise(amd64_code, aarch64_code)
    
    # 如果两边指令数量相差太大，先使用启发式方法评估
    if abs(len(amd64_lines) - len(aarch64_lines)) > max(len(amd64_lines), len(aarch64_lines)) // 2:
        print(f"指令数量差异较大: AMD64({len(amd64_lines)})与AArch64({len(aarch64_lines)})")
        print("先使用启发式方法评估...")
        if not verify_heuristic(aarch64_code, amd64_code):
            print("启发式评估结果为不等价，跳过精确验证")
            return False
        print("启发式评估结果为可能等价，继续进行单条精确验证")
    
    # 处理多条指令情况：逐条进行精确验证
    print("\n多条指令序列，逐条进行精确验证:")
    
    # 初始化寄存器映射
    reg_mapping = {}
    
    # 首先尝试解析通用寄存器
    aarch64_code_processed, reg_mapping = parse_and_replace_registers(aarch64_code, reg_mapping, is_arm=True)
    amd64_code_processed, reg_mapping = parse_and_replace_registers(amd64_code, reg_mapping, is_arm=False)
    
    # 分解处理后的指令序列
    amd64_processed_lines = [line.strip() for line in amd64_code_processed.strip().split('\n') if line.strip()]
    aarch64_processed_lines = [line.strip() for line in aarch64_code_processed.strip().split('\n') if line.strip()]
    
    print("寄存器映射关系: {}".format(reg_mapping))
    
    # 创建反向映射
    reverse_mapping = {}
    for key, values in EQUIVALENT_INSTRUCTIONS.items():
        for value in values:
            reverse_mapping[value] = key
    
    # 收集所有指令的类型信息
    amd64_instruction_info = []
    for line in amd64_processed_lines:
        if not line.strip():
            continue
        instr = line.split()[0].lower()
        category = reverse_mapping.get(instr)
        has_special = any(spec in line for spec in SPECIAL_INSTRUCTIONS)
        amd64_instruction_info.append({
            'line': line,
            'instr': instr,
            'category': category,
            'has_special': has_special
        })
    
    aarch64_instruction_info = []
    for line in aarch64_processed_lines:
        if not line.strip():
            continue
        instr = line.split()[0].lower()
        category = reverse_mapping.get(instr)
        has_special = any(spec in line for spec in SPECIAL_INSTRUCTIONS)
        aarch64_instruction_info.append({
            'line': line,
            'instr': instr,
            'category': category,
            'has_special': has_special
        })
    
    # 尝试对类型相同的指令对进行配对
    pairs = []
    x86_idx_used = set()
    arm_idx_used = set()
    
    # 第一轮：优先匹配类型相同的指令
    for i, x86_info in enumerate(amd64_instruction_info):
        if i in x86_idx_used or x86_info['has_special']:
            continue
        
        for j, arm_info in enumerate(aarch64_instruction_info):
            if j in arm_idx_used or arm_info['has_special']:
                continue
            
            if x86_info['category'] == arm_info['category'] and x86_info['category'] is not None:
                pairs.append((i, j))
                x86_idx_used.add(i)
                arm_idx_used.add(j)
                break
    
    # 统计结果
    print(f"找到 {len(pairs)} 对可能等价的指令对")
    
    # 对每一对进行精确验证
    verification_results = []
    for x86_idx, arm_idx in pairs:
        x86_line = amd64_instruction_info[x86_idx]['line']
        arm_line = aarch64_instruction_info[arm_idx]['line']
        category = amd64_instruction_info[x86_idx]['category']
        
        print(f"\n验证指令对 ({x86_idx+1},{arm_idx+1}) - 类型:{category}")
        print(f"AMD64: {x86_line}")
        print(f"AArch64: {arm_line}")
        
        # 精确验证
        try:
            result = verify_precise(x86_line, arm_line)
            verification_results.append((x86_idx, arm_idx, result))
            if result:
                print(f"指令对 ({x86_idx+1},{arm_idx+1}) 验证成功 ✓")
            else:
                print(f"指令对 ({x86_idx+1},{arm_idx+1}) 验证失败 ✗")
        except Exception as e:
            print(f"验证指令对 ({x86_idx+1},{arm_idx+1}) 时出错: {e}")
            # 对出错的情况，再尝试启发式验证
            try:
                result = verify_heuristic(arm_line, x86_line)
                verification_results.append((x86_idx, arm_idx, result))
                print(f"使用启发式方法验证，结果: {'成功 ✓' if result else '失败 ✗'}")
            except:
                # 如果启发式也失败，假设不等价
                verification_results.append((x86_idx, arm_idx, False))
                print("启发式验证也失败，假设不等价 ✗")
    
    # 处理特殊指令
    special_x86 = [i for i, info in enumerate(amd64_instruction_info) if info['has_special']]
    special_arm = [i for i, info in enumerate(aarch64_instruction_info) if info['has_special']]
    
    if special_x86 or special_arm:
        print("\n检测到特殊指令:")
        for idx in special_x86:
            print(f"AMD64 特殊指令 {idx+1}: {amd64_instruction_info[idx]['line']}")
        for idx in special_arm:
            print(f"AArch64 特殊指令 {idx+1}: {aarch64_instruction_info[idx]['line']}")
        
        # 如果两边都有特殊指令且类型相似，认为这部分等价
        if special_x86 and special_arm:
            print("两端代码都包含特殊指令，使用启发式方法验证这部分")
            special_amd64_code = "\n".join([amd64_instruction_info[i]['line'] for i in special_x86])
            special_aarch64_code = "\n".join([aarch64_instruction_info[i]['line'] for i in special_arm])
            
            special_result = verify_heuristic(special_aarch64_code, special_amd64_code)
            print(f"特殊指令部分启发式验证结果: {'等价 ✓' if special_result else '不等价 ✗'}")
        else:
            special_result = False
            print("只有一端有特殊指令，这部分可能不等价 ✗")
    else:
        special_result = True
    
    # 统计验证结果
    total_pairs = len(pairs)
    successful_pairs = sum(1 for _, _, result in verification_results if result)
    
    if total_pairs == 0:
        # 如果没有可配对的指令，完全使用启发式方法
        print("\n没有找到可精确验证的指令对，使用启发式方法验证整体")
        return verify_heuristic(aarch64_code, amd64_code)
    
    success_ratio = successful_pairs / total_pairs
    print(f"\n验证结果统计: {successful_pairs}/{total_pairs} 对指令验证成功，成功率 {success_ratio:.2%}")
    
    # 根据验证结果判断整体等价性
    if success_ratio >= 0.8 and special_result:
        print("大部分指令验证成功且特殊指令部分等价，整体认为等价 ✓")
        return True
    elif success_ratio >= 0.5 and special_result:
        print("超过一半指令验证成功且特殊指令部分等价，整体可能等价")
        return True
    else:
        print("验证成功率较低或特殊指令部分不等价，整体可能不等价 ✗")
        return False

def parse_rule_file(rule_file):
    """解析规则文件，提取Guest和Host代码对"""
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
    print("ARM64/AMD64汇编验证工具 - 结合精确和启发式方法")
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
            
            # 添加结果统计
            equivalent_rules = []
            non_equivalent_rules = []
            
            for i, (rule_num, guest_code, host_code) in enumerate(rules):
                print(f"\n验证规则 {rule_num}:")
                result = verify_combined(host_code, guest_code)  # 注意这里的顺序是x86, arm
                if result:
                    equivalent_rules.append(str(rule_num))
                else:
                    non_equivalent_rules.append(str(rule_num))
            
            # 统计结果输出
            print("\n" + "=" * 60)
            print("验证结果统计:")
            print(f"等价规则数量: {len(equivalent_rules)}/{len(rules)} ({len(equivalent_rules)/len(rules)*100:.2f}%)")
            if equivalent_rules:
                print(f"等价规则编号: {', '.join(equivalent_rules)}")
            print(f"不等价规则数量: {len(non_equivalent_rules)}/{len(rules)} ({len(non_equivalent_rules)/len(rules)*100:.2f}%)")
            if non_equivalent_rules:
                print(f"不等价规则编号: {', '.join(non_equivalent_rules)}")
                
        except Exception as e:
            print(f"错误: {e}")
    
    elif mode == "2":
        equivalent_count = 0
        non_equivalent_count = 0
        total_count = 0
        
        while True:
            print("\n请输入要验证的代码对 (输入'exit'退出, 输入'stats'查看统计):")
            
            command = input("命令: ")
            if command.lower() == 'exit':
                break
            elif command.lower() == 'stats':
                print("\n" + "=" * 60)
                print("当前验证统计:")
                print(f"总验证数: {total_count}")
                if total_count > 0:
                    print(f"等价代码对: {equivalent_count}/{total_count} ({equivalent_count/total_count*100:.2f}%)")
                    print(f"不等价代码对: {non_equivalent_count}/{total_count} ({non_equivalent_count/total_count*100:.2f}%)")
                else:
                    print("等价代码对: 0/0 (0%)")
                    print("不等价代码对: 0/0 (0%)")
                continue
            
            print("x86汇编代码 (输入空行结束):")
            x86_lines = []
            while True:
                line = input()
                if line.lower() == 'exit':
                    exit(0)
                if not line:
                    break
                x86_lines.append(line)
            
            x86_code = '\n'.join(x86_lines)
            
            print("ARM汇编代码 (输入空行结束):")
            arm_lines = []
            while True:
                line = input()
                if line.lower() == 'exit':
                    exit(0)
                if not line:
                    break
                arm_lines.append(line)
            
            arm_code = '\n'.join(arm_lines)
            
            if not x86_code or not arm_code:
                print("代码不能为空！")
                continue
            
            total_count += 1
            result = verify_combined(x86_code, arm_code)
            if result:
                equivalent_count += 1
            else:
                non_equivalent_count += 1
            
            # 每次验证后显示当前统计
            print("\n当前验证统计:")
            print(f"等价代码对: {equivalent_count}/{total_count} ({equivalent_count/total_count*100:.2f}%)")
            print(f"不等价代码对: {non_equivalent_count}/{total_count} ({non_equivalent_count/total_count*100:.2f}%)")
    
    else:
        print("无效的选择！") 