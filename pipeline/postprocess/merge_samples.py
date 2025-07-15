import json
from pathlib import Path
from collections import Counter

# --- 설정 ---
# 병합할 JSON 파일들의 경로 리스트를 지정하세요.
INPUT_FILES = [
    Path("../result/sampled_results/sampled_50_medgemma-27b_chexpert-plus_sampled200_filtered_structured.json"),
    Path("../result/sampled_results/sampled_50_medgemma-27b_mimic-cxr_sampled200_filtered_structured.json"),
    Path("../result/sampled_results/sampled_50_medgemma-27b_mimic-iv-note_sampled200_filtered_structured.json"),
    Path("../result/sampled_results/sampled_50_medgemma-27b_ReXGradient-160K_sampled200_filtered_structured.json"),
    Path("../result/sampled_results/sampled_50_medgemma-27b_yale_internal_keywords_max3_filtered_structured.json"),
]

MODEL_NAME = "medgemma-27b"
# 병합된 최종 결과물을 저장할 디렉토리 및 기본 파일명
OUTPUT_DIR = Path("../result/final_sample_250")
OUTPUT_FILENAME_BASE = "final_combined"
# ---

def load_json_data(file_path: Path) -> dict:
    """지정된 경로의 JSON 파일을 안전하게 로드합니다."""
    if not file_path.exists():
        print(f"경고: 파일이 존재하지 않아 건너뜁니다 - {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"오류: JSON 형식이 올바르지 않습니다 - {file_path}")
        return None

def main():
    """여러 JSON 결과 파일을 병합하고 두 개의 파일로 나누어 저장하는 메인 함수."""
    print("--- 결과 파일 병합 및 분할 시작 ---")

    all_configs = []
    reports_for_file_A = []
    reports_for_file_B = []
    all_case_ids = []

    # 1. 모든 파일을 순회하며 데이터 수집 및 분할
    for file_path in INPUT_FILES:
        print(f"처리 중: {file_path.name}")
        data = load_json_data(file_path)
        if not data:
            continue

        # 설정(config) 정보 추가
        if "pipeline_configuration" in data:
            all_configs.append(data["pipeline_configuration"])
        else:
            print(f"  경고: '{file_path.name}' 파일에 'pipeline_configuration' 키가 없습니다.")
            
        # 보고서(report) 정보 수집 및 분할
        reports = data.get("all_processed_reports", [])
        if not reports:
            print(f"  정보: '{file_path.name}' 파일에 처리된 보고서가 없습니다.")
            continue
        
        # 전체 case_id를 중복 검사를 위해 먼저 수집
        all_case_ids.extend([report.get('case_id') for report in reports if report.get('case_id')])

        # 현재 파일의 보고서를 절반으로 나눔
        split_index = len(reports) // 2
        reports_part_A = reports[:split_index]
        reports_part_B = reports[split_index:]

        # 각 부분의 보고서를 최종 리스트에 추가
        reports_for_file_A.extend(reports_part_A)
        reports_for_file_B.extend(reports_part_B)

    # 2. Case ID 중복 검사 로직
    if not all_case_ids:
        print("\n처리할 보고서가 없어 중복 검사를 건너뜁니다.")
    else:
        id_counts = Counter(all_case_ids)
        duplicates = {case_id: count for case_id, count in id_counts.items() if count > 1}
        
        print("\n--- case_id 중복 검사 ---")
        if duplicates:
            print("경고: 중복된 case_id가 발견되었습니다!")
            for case_id, count in duplicates.items():
                print(f"  - Case ID '{case_id}' 가 {count}번 나타났습니다.")
        else:
            print("완료: 중복된 case_id가 없습니다.")
        print("--------------------------")

    # 3. 최종 병합된 데이터 구조 생성 (A, B 파일 각각)
    output_data_A = {
        "pipeline_configurations": all_configs,
        "all_processed_reports": reports_for_file_A
    }
    
    output_data_B = {
        "pipeline_configurations": all_configs,
        "all_processed_reports": reports_for_file_B
    }

    # 4. 최종 파일들 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_file_A_path = OUTPUT_DIR / f"{OUTPUT_FILENAME_BASE}_{MODEL_NAME}_part_A.json"
    output_file_B_path = OUTPUT_DIR / f"{OUTPUT_FILENAME_BASE}_{MODEL_NAME}_part_B.json"
    
    with open(output_file_A_path, 'w', encoding='utf-8') as f:
        json.dump(output_data_A, f, indent=2, ensure_ascii=False)
        
    with open(output_file_B_path, 'w', encoding='utf-8') as f:
        json.dump(output_data_B, f, indent=2, ensure_ascii=False)

    print("\n--- 병합 및 분할 완료 ---")
    print(f"총 {len(INPUT_FILES)}개의 파일이 2개의 분할된 파일로 저장되었습니다:")
    print(f"  - 파일 A: '{output_file_A_path}' (보고서 {len(reports_for_file_A)}개)")
    print(f"  - 파일 B: '{output_file_B_path}' (보고서 {len(reports_for_file_B)}개)")
    print(f"두 파일 모두 {len(all_configs)}개의 설정을 포함합니다.")
    print("-------------------------")


if __name__ == "__main__":
    main()