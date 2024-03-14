import csv

# 读取 output.csv 文件
input_file_path = '/home/kingdom/PalmTree-master/angr/test/output.csv'
output_file_path = '/home/kingdom/PalmTree-master/angr/test/output_restored.csv'

with open(input_file_path, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file_path, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ['amd64_code', 'aarch64_code']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    for row in reader:
        amd64_code = row['amd64_code']
        aarch64_code = row['aarch64_code']

        # 替换 amd64_code 中的 0xn000
        parts = amd64_code.split(';')
        restored_parts = []
        label_count = 1
        for part in parts:
            while f"0x{label_count}000" in part:
                part = part.replace(f"0x{label_count}000", f"label{label_count}")
                label_count += 1
            restored_parts.append(part)
        restored_amd64_code = ';'.join(restored_parts)

        # 替换 aarch64_code 中的 0xn000
        parts = aarch64_code.split(';')
        restored_parts = []
        label_count = 1
        for part in parts:
            while f"0x{label_count}000" in part:
                part = part.replace(f"0x{label_count}000", f"label{label_count}")
                label_count += 1
            restored_parts.append(part)
        restored_aarch64_code = ';'.join(restored_parts)

        writer.writerow({'amd64_code': restored_amd64_code, 'aarch64_code': restored_aarch64_code})