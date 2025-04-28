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
            # if state.solver.symbolic(final_sym_val):
                # The value is not symbolic; hence, it's considered constant
                changed_registers[reg] = final_sym_val
    if changed_registers=={}:
        for reg, final_sym_val in final_values.items():
            initial_sym_val = initial_values[reg]
            if str(initial_sym_val) != str(final_sym_val):
                changed_registers[reg] = final_sym_val
    return changed_registers

def execute_assembly(assembly_code, arch, steps):
    project = angr.load_shellcode(assembly_code, arch=arch)
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
    else:
        raise ValueError("不支持的架构类型")
    

    
    registers = set()
    block = project.factory.block(0)
    capstone_instructions = block.capstone.insns
    for ci in capstone_instructions:
        for op in ci.operands:
        # 在这里可以进一步分析操作数op
            if op.type == CS_OP_REG:
                registers.add(ci.reg_name(op.value.reg))
            elif op.type == CS_OP_MEM:
            # 计算并添加内存地址信息
            # 注意：这里的示例不直接适用，因为需要具体架构的上下文来计算地址
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
                        address += getattr(state.regs, str(mem_info['index']))* mem_info['shift_amount']
                    elif 'scale' in mem_info and mem_info['scale']:
                        address += getattr(state.regs, str(mem_info['index']))* mem_info['scale']
                memory_addresses.append(mem(address=address,disp = mem_info['disp'], reg = mem_info['base']
                                            , value = state.memory.load(address, 8, endness=state.arch.memory_endness)))
                if op.value.mem.base != 0:
                    registers.add(mem_info['base'])
                if op.value.mem.index != 0:
                    registers.add(mem_info['index'])

    

    simgr = project.factory.simulation_manager(state)
    # 按照指定的步数执行符号执行
    for _ in range(steps):
        simgr.step()
    # 获取并返回当前状态以及寄存器赋值列表
    if simgr.active:
        return simgr.active, registers_values, memory_addresses, registers
    else:
        print("没有活跃的状态。执行可能未按预期进行。")
        return None, registers_values

def verify_assembly(amd64_code, aarch64_code):
    print("=" * 60)
    print("我想要验证的amd64汇编是:{}".format(amd64_code))
    print("我想要验证的aarch64汇编是:{}".format(aarch64_code))
    print()
    flag = True
    
    # 符号化执行得到最终状态
    final_state_amd64, amd64, mem_amd, amd64_registers = execute_assembly(amd64_code, "amd64", 1)
    final_state_aarch64, aarch64, mem_arm, aarch64_registers = execute_assembly(aarch64_code, "aarch64", 1)
    
    # 循环对x4到x19进行操作
    for i in range(4, 30):
        # 提取63到32位
        x_high_32 = claripy.Extract(63, 32, aarch64[f'x{i}'])
        # 连接成一个64位变量
        x_combined = claripy.Concat(x_high_32, aarch64[f'w{i}'])
        # 将操作后的变量存储到字典中
        aarch64[f'x{i}'] = x_combined
        
    # 解析内存与寄存器
    print("amd寄存器有: ", amd64_registers)
    print("arm寄存器有: ", aarch64_registers)
    
    if len(final_state_amd64) == 1 and len(final_state_aarch64) == 1:
        final_state_amd64 = final_state_amd64[0]
        final_state_aarch64 = final_state_aarch64[0]
        
        amd64_registers_value = get_register_values(final_state_amd64, amd64_registers)
        aarch64_registers_value = get_register_values(final_state_aarch64, aarch64_registers)
        
        registers_output = find_changed_registers(amd64, amd64_registers_value, final_state_amd64)
        arm_registers_output = find_changed_registers(aarch64, aarch64_registers_value, final_state_aarch64)
        
        print("amd_registers_output输出寄存器有: ", registers_output)
        print("arm_registers_output改变的寄存器有: ", arm_registers_output)
        
        mappings = {
            'rax': 'x4', 'rcx': 'x5', 'rdx': 'x6', 'rbx': 'x7',
            'rsp': 'x8', 'rbp': 'x9', 'rsi': 'x10', 'rdi': 'x11',
            'r8': 'x12', 'r9': 'x13', 'r10': 'x14', 'r11': 'x15',
            'r12': 'x16', 'r13': 'x17', 'r14': 'x19', 'r15': 'fp',  
        }
        
        mappings2 = {
            'rax': 'x4', 'rcx': 'x5', 'rdx': 'x6', 'rbx': 'x7',
            'rsp': 'x8', 'rbp': 'x9', 'rsi': 'x10', 'rdi': 'x11',
            'r8': 'x12', 'r9': 'x13', 'r10': 'x14', 'r11': 'x15',
            'r12': 'x16', 'r13': 'x17', 'r14': 'x19', 'r15': 'fp',
            'eax': 'w4', 'ecx': 'w5', 'edx': 'w6', 'ebx': 'w7',
            'esp': 'w8', 'ebp': 'w9', 'esi': 'w10', 'edi': 'w11',
            'r8d': 'w12', 'r9d': 'w13', 'r10d': 'w14', 'r11d': 'w15',
            'r12d': 'w16', 'r13d': 'w17', 'r14d': 'w19', 'r15d': 'w29',    
        }
        
        mappings3 = {
            'ax': 'w4', 'cx': 'w5', 'dx': 'w6', 'bx': 'w7',
            'sp': 'w8', 'bp': 'w9', 'si': 'w10', 'di': 'w11',
            'r8w': 'w12', 'r9w': 'w13', 'r10w': 'w14', 'r11w': 'w15',
            'r12w': 'w16', 'r13w': 'w17', 'r14w': 'w19', 'r15w': 'w29', 
        }
        
        mappings4 = {
            'al': 'w4', 'cl': 'w5', 'dl': 'w6', 'bl': 'w7',
            'spl': 'w8', 'bpl': 'w9', 'sil': 'w10', 'dil': 'w11',
            'r8b': 'w12', 'r9b': 'w13', 'r10b': 'w14', 'r11b': 'w15',
            'r12b': 'w16', 'r13b': 'w17', 'r14b': 'w19', 'r15b': 'w29', 
        }
        
        s = claripy.Solver()
        
        for amd64_reg, aarch64_reg in mappings.items():
            s.add(amd64[amd64_reg] == aarch64[aarch64_reg])
            
        for mem_amd_obj in mem_amd:
            for mem_arm_obj in mem_arm:
                if mem_amd_obj.disp == mem_arm_obj.disp and mem_amd_obj.reg in mappings and mem_arm_obj.reg == mappings[mem_amd_obj.reg]:
                    memory_content_amd1 = mem_amd_obj.value
                    memory_content_arm1 = mem_arm_obj.value
                    memory_content_amd2 = final_state_amd64.memory.load(mem_amd_obj.address, 8, endness=final_state_amd64.arch.memory_endness)
                    memory_content_arm2 = final_state_aarch64.memory.load(mem_arm_obj.address, 8, endness=final_state_aarch64.arch.memory_endness)
                    
                    print("初始amd内存是", memory_content_amd1)
                    print("初始arm内存是", memory_content_arm1)
                    print("最终amd内存是", memory_content_amd2)
                    print("最终arm内存是", memory_content_arm2)
                    
                    s.add(memory_content_amd1 == memory_content_arm1)
                    
                    if (str(memory_content_amd1) != str(memory_content_amd2)):
                        if s.satisfiable(extra_constraints=[memory_content_amd2 != memory_content_arm2]):
                            flag = False

        for reg in registers_output:
            if reg in mappings2:
                arm_reg = mappings2.get(reg)
                arm_expr = aarch64_registers_value[arm_reg]
                amd_expr = amd64_registers_value[reg]

                if s.satisfiable(extra_constraints=[arm_expr != amd_expr]):
                    flag = False

            if reg in mappings3:
                arm_reg = mappings3.get(reg)        
                arm_expr = claripy.Extract(15, 0, aarch64_registers_value[arm_reg])
                amd_expr = amd64_registers_value[reg]
                
                if s.satisfiable(extra_constraints=[arm_expr != amd_expr]):
                    flag = False

            if reg in mappings4:
                arm_reg = mappings4.get(reg)
                arm_expr = claripy.Extract(7, 0, aarch64_registers_value[arm_reg])
                amd_expr = amd64_registers_value[reg]

                if s.satisfiable(extra_constraints=[arm_expr != amd_expr]):
                    flag = False
                    
        if flag:
            print("两段汇编只有一条路径，且两段汇编等价")
            return True
        else:
            print("两段汇编不等价")
            return False
            
    elif len(final_state_amd64) > 1 and len(final_state_aarch64) > 1 and len(final_state_amd64) == len(final_state_aarch64):
        # 多个执行路径的处理逻辑...
        print("汇编代码有多个执行路径，等价性验证较复杂")
        return None
    else:
        print("两端汇编的执行路径数量不同，不等价")
        return False

if __name__ == "__main__":
    # 示例用法
    verify_assembly("mov rax, 42", "mov x4, #42")
    verify_assembly("add rax, 10", "add x4, x4, #10")
    
    # 接受用户输入进行验证
    while True:
        print("\n请输入要验证的汇编代码对（输入'exit'退出）：")
        amd64_input = input("X86汇编代码: ")
        if amd64_input.lower() == 'exit':
            break
        
        aarch64_input = input("ARM汇编代码: ")
        if aarch64_input.lower() == 'exit':
            break
        
        verify_assembly(amd64_input, aarch64_input) 