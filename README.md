# Đánh giá mô hình Text-to-Speech tiếng Việt

## Thành viên nhóm

1. Nguyễn Huy Hoàng - 23122031
2. Trần Tạ Quang Minh - 23122042
3. Nguyễn Bá Nam - 23122043
4. Lâm Hoàng Vũ - 23122056

## Mô tả

Phần Đánh giá mô hình - Đồ án triển khai TTS trên thiết bị di động 

## Cấu trúc dự án

```
NLP-CK/
├── ground-truth/          # Văn bản gốc (ground truth)
├── xtts/                  # Output từ XTTS model
├── f5tts-base/           # Output từ F5-TTS base  
├── f5tts-ours/           # Output từ F5-TTS custom model
├── evaluate_wer.py        # Tính WER giữa các model
├── analyze_wer.py         # Phân tích thống kê WER
├── [EVAL] XTTS.ipynb     # Pipeline XTTS
└── [EVAL] F5-TTS.ipynb      # Pipeline F5-TTS
```

## Scripts chính

### 1. Extract Transcript
Trích xuất văn bản từ file audio sử dụng Whisper ASR.

```bash
python extract_transcript.py -i audio_folder -o output_folder -n 100
```

### 2. Evaluate WER
Output:ord Error Rate giữa ground-truth và output của các model.

```bash
# Single model
python evaluate_wer.py -g ground-truth -m xtts/text -n XTTS -o wer_results

# Multiple models
python evaluate_wer.py -g ground-truth \
  -m xtts/text f5tts-100k f5tts-500k \
  -n XTTS F5-TTS-100k F5-TTS-500k \
  -o wer_results
```

**Output:** CSV file với các cột `id`, `ground_truth`, `[model_name]`, `wer`

### 3. Analyze WER Statistics
Phân tích thống kê chi tiết với các metrics về độ tin cậy.

```bash
python analyze_wer.py -i wer_results -o wer_analysis
```

Metrics bao gồm:
- Cơ bản: Mean, Median, Std, Min, Max
- Phân phối: Q1, Q3, IQR, Skewness, Kurtosis
- Độ tin cậy: Coefficient of Variation, SEM, 95% CI
- Phân loại: Excellent/Good/Fair/Poor performance

Output:
- `model_comparison.csv` - So sánh các model
- `wer_analysis_report.txt` - Báo cáo chi tiết
- `statistics.json` - Dữ liệu JSON

## Notebooks

- [EVAL]_XTTS.ipynb: Pipeline XTTS - Text → Audio → Whisper → Transcript
- eval-f5tts.ipynb: Pipeline F5-TTS - Text → Audio → Whisper → Transcript

## Pipeline đánh giá

```
1. Ground-truth (text files)
   ↓
2. TTS Model (XTTS/F5-TTS)
   ↓
3. Audio files (.wav)
   ↓
4. Whisper ASR
   ↓
5. Transcribed text
   ↓
6. WER Evaluation
   ↓
7. Statistical Analysis
```

## Kết quả

Kết quả đánh giá được lưu trong các folder:
- `wer_results/` - CSV files chứa WER cho từng model
- `wer_analysis/` - Báo cáo thống kê chi tiết và so sánh

## Dependencies

```bash
pip install jiwer pandas numpy soundfile openai-whisper
```