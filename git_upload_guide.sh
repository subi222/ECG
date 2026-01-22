#!/bin/bash
# GitHub 업로드 가이드

cd /home/subi/VSCodeProjects/ECG

echo "=== Step 1: .gitignore 수정 ==="
# benchmark_results_for_github 폴더만 예외로 추가
cat >> .gitignore << 'EOF'

# Benchmark results for GitHub (exception)
!outputs/benchmark_results_for_github/
!outputs/benchmark_results_for_github/**
EOF

echo "✅ .gitignore 수정 완료"
echo ""

echo "=== Step 2: Git 상태 확인 ==="
git status outputs/benchmark_results_for_github/ | head -20
echo ""

echo "=== Step 3: 파일 추가 ==="
echo "다음 명령어를 실행하세요:"
echo ""
echo "git add -f outputs/benchmark_results_for_github/"
echo "git commit -m 'Add benchmark results (CSV + 0dB plots for BW/EM/MA)'"
echo "git push"
echo ""

echo "=== 업로드 내용 ==="
echo "- CSV 파일: 3개 (BW, EM, MA summary)"
echo "- Plot 파일: 84개 (0dB SNR only)"
echo "- 전체 크기: 26MB"
echo "- 모델: Proposed, DeepFilter, FCN+DAE, DRNN"
