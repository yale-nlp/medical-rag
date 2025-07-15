import json
import random
from pathlib import Path
import copy

# --- 설정 ---
# 비교할 두 개의 JSON 파일 경로를 지정하세요.
FILE_1_PATH = Path("/home/dj475/yalenlp/medical-rag/250616/result/llama3.3-70b/yale_internal/yale_internal_keywords_max3_filtered_structured.json") 
FILE_2_PATH = Path("/home/dj475/yalenlp/medical-rag/250616/result/medgemma-27b/yale_internal/yale_internal_keywords_max3_filtered_structured.json") 

# 샘플링된 결과 파일을 저장할 디렉토리
OUTPUT_DIR = Path("../result/sampled_results")

MODEL1 = "llama3.3-70b"
MODEL2 = "medgemma-27b"

# 샘플링할 케이스 개수
NUM_SAMPLES = 50
# -----------

def load_json_data(file_path: Path) -> dict:
    """지정된 경로의 JSON 파일을 안전하게 로드합니다."""
    if not file_path.exists():
        print(f"오류: 파일이 존재하지 않습니다 - {file_path}")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"오류: JSON 형식이 올바르지 않습니다 - {file_path}")
        return None

def main():
    """메인 비교 및 샘플링 로직을 수행합니다."""
    print(f"1번 파일: {FILE_1_PATH}")
    print(f"2번 파일: {FILE_2_PATH}")

    data1 = load_json_data(FILE_1_PATH)
    data2 = load_json_data(FILE_2_PATH)

    if not data1 or not data2:
        print("파일 로딩 실패. 스크립트를 종료합니다.")
        return

    # 각 파일에서 case_id 목록을 추출합니다.
    try:
        reports1 = data1.get("all_processed_reports", [])
        reports2 = data2.get("all_processed_reports", [])
        
        case_ids1 = {report['case_id'] for report in reports1}
        case_ids2 = {report['case_id'] for report in reports2}
    except (KeyError, TypeError) as e:
        print(f"오류: JSON 파일 구조가 예상과 다릅니다. 'all_processed_reports' 또는 'case_id' 키를 찾을 수 없습니다. ({e})")
        return

    if not case_ids1:
        print(f"오류: 1번 파일 '{FILE_1_PATH.name}'에 처리된 케이스가 없습니다.")
        return

    # 일치하는 case_id를 찾고 통계를 계산합니다.
    matching_ids = case_ids1.intersection(case_ids2)
    
    num_total = len(case_ids1)
    num_matching = len(matching_ids)
    ratio = num_matching / num_total if num_total > 0 else 0

    print("\n--- 비교 결과 ---")
    print(f"전체 케이스({num_total}개) 대비 일치하는 케이스 수: {num_matching}개 ({ratio:.2%})")

    if not matching_ids:
        print("일치하는 케이스가 없어 샘플링을 진행하지 않습니다.")
        return

    # 일치하는 케이스 중 50개 (또는 그 이하)를 랜덤 샘플링합니다.
    num_to_sample = min(NUM_SAMPLES, num_matching)
    sampled_ids_set = set(random.sample(list(matching_ids), num_to_sample))
    
    print(f"\n--- 샘플링 ---")
    print(f"일치하는 케이스 {num_matching}개 중 {num_to_sample}개를 랜덤 샘플링하여 2개의 새 파일을 생성합니다...")

    # 샘플링된 결과 파일을 저장할 디렉토리를 생성합니다.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 첫 번째 파일 기반의 샘플링된 결과 생성
    sampled_reports_1 = [report for report in reports1 if report['case_id'] in sampled_ids_set]
    output_data_1 = copy.deepcopy(data1)
    output_data_1["all_processed_reports"] = sampled_reports_1
    
    output_file_1_path = OUTPUT_DIR / f"sampled_50_{MODEL1}_{FILE_1_PATH.stem}.json"
    with open(output_file_1_path, 'w', encoding='utf-8') as f:
        json.dump(output_data_1, f, indent=2, ensure_ascii=False)

    # 2. 두 번째 파일 기반의 샘플링된 결과 생성
    sampled_reports_2 = [report for report in reports2 if report['case_id'] in sampled_ids_set]
    output_data_2 = copy.deepcopy(data2)
    output_data_2["all_processed_reports"] = sampled_reports_2
    
    output_file_2_path = OUTPUT_DIR / f"sampled_50_{MODEL2}_{FILE_2_PATH.stem}.json"
    with open(output_file_2_path, 'w', encoding='utf-8') as f:
        json.dump(output_data_2, f, indent=2, ensure_ascii=False)

    print("\n--- 완료 ---")
    print(f"{num_to_sample}개의 샘플 케이스를 포함한 2개의 결과 파일이 '{OUTPUT_DIR}' 디렉토리에 저장되었습니다:")
    print(f"  - {output_file_1_path.name}")
    print(f"  - {output_file_2_path.name}")


if __name__ == "__main__":
    main()