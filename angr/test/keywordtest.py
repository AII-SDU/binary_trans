import pandas as pd

def preprocess_aarch64(aarch64_code):
    """
    Remove duplicate and adrp instructions from aarch64_code, ensuring input is treated as string.
    """
    # Ensure aarch64_code is treated as a string
    aarch64_code_str = str(aarch64_code)
    # Split the code into instructions and remove duplicates
    instructions = list(dict.fromkeys(aarch64_code_str.split(';')))
    # Remove any instruction that contains 'adrp'
    instructions = [insn for insn in instructions if 'adrp' not in insn]
    return ';'.join(instructions)

def translate_operand(operand):
    """
    Translate an operand for ARM architecture, adding '#' before hex values.
    """
    if '0x' in operand:
        return '#' + operand
    return operand

def translate_amd64_instruction(insn):
    """
    Translate a single AMD64 instruction to AARCH64 based on provided rules.
    """
    if insn.startswith('push'):
        operands = insn.replace('push ', '')
        return f"str {translate_operand(operands)}, [x8, #-8]!"
    elif insn.startswith('pop'):
        operands = insn.replace('pop ', '')
        return f"ldr {translate_operand(operands)}, [x8];add x8, x8, #8"
    elif insn.startswith('add rsp,'):
        value = insn.replace('add rsp,', '').strip()
        return f"add x8, x8, {translate_operand(value)}"
    elif insn.startswith('sub rsp,'):
        value = insn.replace('sub rsp,', '').strip()
        return f"sub x8, x8, {translate_operand(value)}"
    else:
        # If the instruction does not match any of the translation rules, return None
        return None

def translate_and_clean_amd64_aarch64(row):
    """
    Translate the first AMD64 instruction if necessary and clean the AARCH64 instructions
    based on the rules provided by the user.
    """
    # Clean aarch64_code first
    cleaned_aarch64 = preprocess_aarch64(row['aarch64_code'])
    
    # Split amd64_code to get the first instruction
    amd64_insns = row['amd64_code'].split(';')
    first_amd64_insn = amd64_insns[0]
    
    # Translate the first AMD64 instruction
    translated_insn = translate_amd64_instruction(first_amd64_insn)
    
    # If the first instruction was translated, prepend it to the cleaned aarch64 code
    if translated_insn:
        cleaned_aarch64 = translated_insn + ';' + cleaned_aarch64
    
    # Update the dataframe row
    return pd.Series([row['amd64_code'], cleaned_aarch64])

# Load the CSV file
file_path = '/home/kingdom/PalmTree-master/angr/test/output_restored.csv'
df = pd.read_csv(file_path)

# Apply the translation and cleaning to each row with corrected handling
df_cleaned = df.apply(translate_and_clean_amd64_aarch64, axis=1)
df_cleaned.columns = ['amd64_code', 'aarch64_code']

# Optionally, save the cleaned dataframe back to a new CSV
output_path = '/home/kingdom/PalmTree-master/angr/test/output_final.csv'
df_cleaned.to_csv(output_path, index=False)


