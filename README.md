# 프로젝트 설명

이 프로젝트는 **공공데이터 및 인천공항의 항공편 정보를 활용**하여 이륙 가능성을 예측하고,  이를 **FastAPI**로 제공하는 웹 서비스입니다.  

## 나의 역할 및 작업 내용

### 1. **데이터 전처리**
- **항공편 출발 정보**(엑셀 파일)와 **기상 데이터**(API 및 CSV)를 **병합**하여 최종적으로 예측에 사용할 **데이터셋**을 생성했습니다.
- 데이터 병합 후, 예측에 필요한 변수들을 **전처리**하고 **feature engineering**을 적용하여 모델 학습에 적합한 형태로 데이터를 변환했습니다.

### 2. **모델 학습**
- **Random Forest**와 **XGBoost** 모델을 사용하여 **이륙 가능성 예측** 모델을 학습했습니다.
- 데이터의 **불균형 문제**를 해결하기 위해 **언더샘플링**을 적용하여 모델 성능을 향상시켰습니다.
- 학습된 모델 파일을 **`.pkl`** 형식으로 저장하여 API에서 사용할 수 있도록 했습니다.

### 3. **API 구현**
- **FastAPI**를 사용하여 **예측 API**와 **기상 정보 API**를 구현했습니다. 사용자는 예측 API를 통해 항공편의 이륙 가능성을 예측할 수 있습니다.
- **기상 정보 API**는 실시간 기상 정보를 제공하여 예측에 영향을 미칠 수 있는 기상 요소를 함께 고려할 수 있게 했습니다.

## 설치 및 실행 방법

### 1. 필수 라이브러리 설치:
먼저 필요한 라이브러리들을 설치해야 합니다. 프로젝트 루트 디렉토리에서 아래 명령어를 실행해주세요:

```bash
pip install -r requirements.txt

```

### 2. 데이터 다운로드:
프로젝트에서 사용하는 데이터는 **항공편 출발 정보**와 **기상 데이터**입니다. 데이터 파일은 **[항공정보포탈]**과 **[기상 정보 API]**에서 다운로드할 수 있습니다.

1. **항공편 출발 정보**:
   - [항공정보포탈 링크](https://www.airportal.go.kr/airport/aircraftInfo.do)에서 **2024년 1월부터 12월까지의 항공편 출발 데이터를** 다운로드하세요.
   - 파일을 `data/takeoff/initial_takeoff_data/initial_takeoff_data` 폴더에 넣으세요.

2. **기상 정보**:
   - [공공데이터포털](https://www.data.go.kr/data/15095109/openapi.do)에서 **기상 정보를** API를 통해 수집합니다.
   - 초기 기상 데이터 파일은  **`collect_weather.ipynb`** 를 통해 `data/weather/initial_weather_data` 폴더에 `weather_data.csv`로 생성됩니다.
   - `data/weather/additional_weather_data` 폴더의  **`additional_weather.ipynb`** 파일을 통해 추가로 수집하고  **`preprocessing_weather.ipynb`** 파일을 통해 전처리를 수행합니다. 이후 `weather_data_fixed.csv` 가 생성됩니다.

### 3. 데이터 병합:
항공편 출발 정보 데이터 다운로드 후 **병합**을 진행해야 합니다. `data/takeoff/total_takeoff_data/` 폴더에 있는 **`merge_initial_takeoff_data.ipynb`** 파일을 실행하여 데이터를 병합합니다.

병합 후, **`data/takeoff/initial_takeoff_data/`** 폴더에 `2024_merge_takeoff.csv` 파일이 생성됩니다.

이후, **`data/takeoff/total_takeoff_data/`** 폴더에 있는 **`preprocessing_takeoff.ipynb`** 파일을 실행하여 전처리를 수행합니다. 수행 후, 해당 폴더에 `2024_takeoff.csv` 파일이 생성됩니다.

기상정보와 항공기 출발편 정보 데이터를 병합합니다. 
**`data/merged_data/data_merge/`** 폴더의 **`data_merge.ipynb`** 파일을 통해 병합합니다. 
**`data/merged_data/preprocess/`** 폴더의 **`preprocessing_merged_data.ipynb`** 파일을 통해 전처리를 수행합니다. 

### 4. FastAPI 서버 실행:
데이터가 준비되면, 아래 명령어로 FastAPI 서버를 실행할 수 있습니다:

```bash
uvicorn main:app --reload

```

## 사용 예시

- **항공편 예측 API**: `/predict_delay` 엔드포인트로 항공편 예측을 요청할 수 있습니다.

- **기상 정보 API**: `/get_weather` 엔드포인트로 현재 공항의 기상 정보를 확인할 수 있습니다.

## 주의 사항

- 데이터 파일을 **다운로드 후 병합**해야만 모델 학습 및 예측이 가능합니다. 데이터 파일은 제공되지 않으므로, **해당 링크에서 직접 다운로드**해야 합니다.






