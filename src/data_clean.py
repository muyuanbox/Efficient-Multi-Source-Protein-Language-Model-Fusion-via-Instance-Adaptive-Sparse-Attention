#!/usr/bin/env python3
"""
在TSV文件的每行添加序列序号
用法: python add_sequence_id.py input.tsv output.tsv [--prefix Sequence] [--start 0]
"""

import argparse


def add_sequence_id(input_file, output_file, prefix="Sequence", start_id=0):
    """
    给TSV文件添加序列序号列
    
    Args:
        input_file: 输入TSV文件路径
        output_file: 输出TSV文件路径
        prefix: 序号前缀（默认为 "Sequence"）
        start_id: 起始序号（默认从0开始）
    """
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for idx, line in enumerate(fin):
            line = line.rstrip('\n\r')
            seq_id = f"{prefix}{idx + start_id}"
            fout.write(f"{seq_id}\t{line}\n")
    
    print(f"完成! 已将结果保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='在TSV文件的每行添加序列序号')
    parser.add_argument('input', help='输入TSV文件路径',default='data/gfp/gfp_sequences.tsv')
    parser.add_argument('output', help='输出TSV文件路径',default='data/gfp/location_with_sequence_id.tsv')
    parser.add_argument('--prefix', default='Sequence', help='序号前缀（默认为 "Sequence"）')
    parser.add_argument('--start', type=int, default=0, help='起始序号（默认为0）')
    
    args = parser.parse_args()
    
    add_sequence_id(args.input, args.output, args.prefix, args.start)


if __name__ == '__main__':
    main()