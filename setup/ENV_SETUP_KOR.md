# Python 환경 설정 가이드

이 문서는 초보 개발자분을 위해 프로젝트 환경을 설정하는 방법을 설명합니다.

## 1. 전제 조건
- **Python 3.12** 버전이 설치되어 있어야 합니다.

## 2. 가상환경 생성 및 라이브러리 설치

터미널(또는 명령 프롬프트)을 열고 프로젝트 루트 폴더에서 다음 순서대로 입력하세요.

### Windows
```bash
# 1. 가상환경 생성
python -m venv .venv

# 2. 가상환경 활성화
.venv\Scripts\activate

# 3. 필수 라이브러리 설치
pip install -r setup/requirements.txt
```

### macOS / Linux
```bash
# 1. 가상환경 생성
python3 -m venv .venv

# 2. 가상환경 활성화
source .venv/bin/activate

# 3. 필수 라이브러리 설치
pip install -r setup/requirements.txt
```

## 3. 실험 실행
환경 설정이 완료되면 다음 명령어로 테스트를 실행할 수 있습니다.
```bash
python run_synthetic_test.py
```

## 주의사항
- **데이터 폴더**: `MITDB_data`, `noise_data` 등은 Git에 이미 포함되어 있으므로 별도로 다운로드할 필요가 없습니다.
- **가상환경**: 코드를 수정하거나 실행하기 전에는 항상 가상환경이 활성화(`.venv` 표시)되어 있는지 확인하세요.
