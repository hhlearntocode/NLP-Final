#!/usr/bin/env python3
"""
Script để trích xuất transcript từ file txt vào folder ground-truth
Chỉ trích xuất text ở cột giữa (giữa 3 cột)
"""

import os
import argparse


def extract_transcript(input_file, output_dir, num_lines=None):
    """
    Trích xuất transcript từ file txt, mỗi dòng thành 1 file riêng
    
    Args:
        input_file: Đường dẫn file input
        output_dir: Thư mục output (ground-truth)
        num_lines: Số dòng cần trích xuất (None = tất cả)
    """
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc và trích xuất
    with open(input_file, 'r', encoding='utf-8') as f_in:
        file_counter = 1
        lines_processed = 0
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            # Tách chuỗi theo dấu |
            parts = line.split('|')
            
            if len(parts) >= 2:
                # Lấy cột giữa (text transcript)
                transcript = parts[1].strip()
                
                # Tạo file riêng cho mỗi transcript
                output_file = os.path.join(output_dir, f"{file_counter}.txt")
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    f_out.write(transcript)
                
                file_counter += 1
                lines_processed += 1
                
                # Kiểm tra số dòng đã xử lý
                if num_lines and lines_processed >= num_lines:
                    break
    
    print(f"✓ Đã trích xuất {lines_processed} dòng transcript")
    print(f"✓ Tạo {lines_processed} files từ 1.txt đến {lines_processed}.txt")
    print(f"✓ Lưu vào folder: {output_dir}/")
    return lines_processed


def main():
    parser = argparse.ArgumentParser(
        description='Trích xuất transcript từ file txt vào folder ground-truth'
    )
    parser.add_argument(
        '-i', '--input',
        default='transcriptAll.txt',
        help='File input (mặc định: transcriptAll.txt)'
    )
    parser.add_argument(
        '-o', '--output',
        default='ground-truth',
        help='Folder output (mặc định: ground-truth)'
    )
    parser.add_argument(
        '-n', '--num-lines',
        type=int,
        help='Số dòng cần trích xuất (mặc định: tất cả)'
    )
    
    args = parser.parse_args()
    
    # Kiểm tra file input tồn tại
    if not os.path.exists(args.input):
        print(f"✗ Lỗi: Không tìm thấy file {args.input}")
        return
    
    # Trích xuất transcript
    extract_transcript(args.input, args.output, args.num_lines)


if __name__ == '__main__':
    main()
