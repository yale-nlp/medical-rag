#!/bin/bash
echo "======================================================"
echo ">> STARTING EVALUTAION 1: "LLAMA4-CHEXPERT"
echo "======================================================"
python run_annotation.py --input_file ../result/llama4/chexpert_2000/chexpert-plus_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/chexpert-plus/llama4 --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 2: "LLAMA4-MIMIC-CXR"
echo "======================================================"
python run_annotation.py --input_file ../result/llama4/mimic-cxr_2000/mimic-cxr_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/mimic-cxr/llama4 --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 3: "LLAMA4-MIMIC-IV"
echo "======================================================"
python run_annotation.py --input_file ../result/llama4/mimic-iv_2000/mimic-iv-note_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/mimic-iv/llama4 --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 4: "LLAMA4-REXGRADIENT"
echo "======================================================"
python run_annotation.py --input_file ../result/llama4/rexgradient_2000/ReXGradient-160K_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/rexgradient/llama4 --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 5: "LLAMA4-YALE-INTERNAL"
echo "======================================================"
python run_annotation.py --input_file ../result/llama4/yale_internal/yale_internal_keywords_max3_local_vllm_feedback.json --output_dir result/huggingface/medgemma/yaleinternal/llama4 --config_file configs/medgemma.json

echo "======================================================"
echo ">> EVALUATION FOR LLAMA4 ENDED"
echo "======================================================"
echo ">> EVALUATION FOR MEDGEMMA START"
echo "======================================================"


echo "======================================================"
echo ">> STARTING EVALUTAION 1: "MEDGEMMA-CHEXPERT"
echo "======================================================"
python run_annotation.py --input_file ../result/medgemma-27b/chexpert_2000/chexpert-plus_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/chexpert-plus/medgemma-27b --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 2: "MEDGEMMA-MIMIC-CXR"
echo "======================================================"
python run_annotation.py --input_file ../result/medgemma-27b/mimic-cxr_2000/mimic-cxr_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/mimic-cxr/medgemma-27b --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 3: "MEDGEMMA-MIMIC-IV"
echo "======================================================"
python run_annotation.py --input_file ../result/medgemma-27b/mimic-iv_2000/mimic-iv-note_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/mimic-iv/medgemma-27b --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 4: "MEDGEMMA-REXGRADIENT"
echo "======================================================"
python run_annotation.py --input_file ../result/medgemma-27b/rexgradient_2000/ReXGradient-160K_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/rexgradient/medgemma-27b --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 5: "MEDGEMMA-YALE-INTERNAL"
echo "======================================================"
python run_annotation.py --input_file ../result/medgemma-27b/yale_internal/yale_internal_keywords_max3_local_vllm_feedback.json --output_dir result/huggingface/medgemma/yaleinternal/medgemma-27b --config_file configs/medgemma.json

echo "======================================================"
echo ">> EVALUATION FOR MEDGEMMA ENDED"
echo "======================================================"
echo ">> EVALUATION FOR QWEN3-32B START"
echo "======================================================"

echo "======================================================"
echo ">> STARTING EVALUTAION 1: "QWEN3-32B-CHEXPERT"
echo "======================================================"
python run_annotation.py --input_file ../result/qwen3-32b/chexpert_2000/chexpert-plus_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/chexpert-plus/qwen3-32b --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 2: "QWEN3-32B-MIMIC-CXR"
echo "======================================================"
python run_annotation.py --input_file ../result/qwen3-32b/mimic-cxr_2000/mimic-cxr_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/mimic-cxr/qwen3-32b --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 3: "QWEN3-32B-MIMIC-IV"
echo "======================================================"
python run_annotation.py --input_file ../result/qwen3-32b/mimic-iv_2000/mimic-iv-note_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/mimic-iv/qwen3-32b --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 4: "QWEN3-32B-REXGRADIENT"
echo "======================================================"
python run_annotation.py --input_file ../result/qwen3-32b/rexgradient_2000/ReXGradient-160K_sampled2000_local_vllm_feedback.json --output_dir result/huggingface/medgemma/rexgradient/qwen3-32b --config_file configs/medgemma.json

echo "======================================================"
echo ">> STARTING EVALUTAION 5: "QWEN3-32B-YALE-INTERNAL"
echo "======================================================"
python run_annotation.py --input_file ../result/qwen3-32b/yale_internal/yale_internal_keywords_max3_local_vllm_feedback.json --output_dir result/huggingface/medgemma/yaleinternal/qwen3-32b --config_file configs/medgemma.json


echo "\n\n"
echo ">> ALL EVALUATIONS COMPLETE."