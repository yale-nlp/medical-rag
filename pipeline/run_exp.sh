#!/bin/bash
echo "==========================================================="
echo ">> STARTING EXPERIMENT 1: LLAMA3.1-8B YALE-INTERNAL"
echo "==========================================================="
python async_main.py --config configs/llama3.1_8b_yale_internal.json

# echo "\n\n"
# echo "==========================================================="
# echo ">> STARTING EXPERIMENT 2: LLAMA3.1-8B MIMIC-IV-2000"
# echo "==========================================================="
# python async_main.py --config configs/llama3.1_8b_mimic-iv_2000.json

echo "\n\n"
echo "==========================================================="
echo ">> STARTING EXPERIMENT 3: LLAMA3.1-8B MIMIC-CXR-2000"
echo "==========================================================="
python async_main.py --config configs/llama3.1_8b_mimic-cxr_2000.json

echo "\n\n"
echo "==========================================================="
echo ">> STARTING EXPERIMENT 4: LLAMA3.1-8B REXGRADIENT-2000"
echo "==========================================================="
python async_main.py --config configs/llama3.1_8b_rexgradient_2000.json

echo "\n\n"
echo "==========================================================="
echo ">> STARTING EXPERIMENT 5: LLAMA3.1-8B CHEXPERT"
echo "==========================================================="
python async_main.py --config configs/llama3.1_8b_chexpert2000.json

echo "\n\n"
echo ">> ALL EXPERIMENTS COMPLETE."