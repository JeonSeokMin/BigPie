from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from typing import List, Dict
import joblib
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# .env 파일 로드
load_dotenv()

# API 키 가져오기 (환경 변수에서)
SERVICE_KEY = os.getenv("SERVICE_KEY")
SERVICE_KEY_2 = os.getenv("SERVICE_KEY_2")

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://big-fly.netlify.app",  # 실제 배포된 프론트엔드 도메인
        "http://localhost:5173",        # 로컬 개발 환경에서의 프론트엔드 도메인
    ],  
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

###########################################################################################################################################################
## 항공편 이륙 가능성 예측 ##
###########################################################################################################################################################

# 모델 로드
model = joblib.load("./predict_delay/ML/with_undersampling/trained_model/random_forest_undersampled_model.pkl")

# 입력 데이터 모델 정의
class FlightInput(BaseModel):
    year: int
    month: int
    day: int
    airline: str
    flight_number: str
    departure_time: str

# ✅ 응답 데이터 모델 (유연한 JSON 응답 유지)
class PredictionOutput(BaseModel):
    prediction_probabilities: Dict[str, float] = Field(
        ..., 
        example={"departure": 43.0, "delay": 48.0, "cancellation": 9.0, "return": 0.0}
    )

# 항공사 및 편명 숫자로 변환
def parse_flight_info(airline, flight_number):
    use_data = pd.read_csv('./predict_delay/data/merged_data/data_merge/merged_data.csv')
    airline_mapping = {name: code for code, name in enumerate(use_data['항공사'].astype('category').cat.categories)}
    flight_mapping = {name: code for code, name in enumerate(use_data['편명'].astype('category').cat.categories)}
    
    if airline not in airline_mapping:
        airline_mapping[airline] = max(airline_mapping.values(), default=0) + 1
    if flight_number not in flight_mapping:
        flight_mapping[flight_number] = max(flight_mapping.values(), default=0) + 1

    return airline_mapping[airline], flight_mapping[flight_number]

# 시간 변환 (YYYYMMDDHH00)
def format_datetime_for_api_fixed_minutes(year, month, day, departure_time):
    hour, _ = map(int, departure_time.split(":"))
    return f"{year}{month:02d}{day:02d}{hour:02d}00", hour

# 기상 정보 API 호출
def get_weather_info(year, month, day, departure_time):
    fctm, hour = format_datetime_for_api_fixed_minutes(year, month, day, departure_time)
    
    url = 'http://apis.data.go.kr/1360000/AirInfoService/getAirInfo'
    params = {
        'serviceKey' : SERVICE_KEY,
        'numOfRows': '100',
        'pageNo': '1',
        'dataType': 'JSON',
        'fctm': fctm,
        'icaoCode': 'RKSI'
    }

    response = requests.get(url, params=params)
    try:
        data = response.json()
        if 'response' in data and 'body' in data['response']:
            items = data['response']['body']['items']['item']
            for item in items:
                if item['tmFc'] == fctm:
                    return item['wd'], item['ws'], item['ta'], item['qnh'], hour
    except Exception as e:
        print("API 호출 중 오류 발생:", e)

    return None, None, None, None, hour

# 사용자 입력 변환
def transform_user_input(user_input):
    airline_id, flight_id = parse_flight_info(user_input.airline, user_input.flight_number)
    fctm, hour = format_datetime_for_api_fixed_minutes(user_input.year, user_input.month, user_input.day, user_input.departure_time)
    minutes = hour * 60 + int(user_input.departure_time.split(":")[1])
    time_slot = pd.cut([hour], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], right=False)[0]
    weekday = pd.Timestamp(user_input.year, user_input.month, user_input.day).dayofweek
    season = (user_input.month % 12) // 3
    wind_dir, wind_speed, temp, pressure, hour = get_weather_info(user_input.year, user_input.month, user_input.day, user_input.departure_time)

    transformed_data = pd.DataFrame([[
        airline_id, flight_id, int(wind_dir) if wind_dir else 180, int(wind_speed) if wind_speed else 5,
        int(temp) if temp else 20, int(pressure) if pressure else 1013, weekday, season, hour,
        int(time_slot), user_input.year, user_input.month, user_input.day, minutes
    ]], columns=model.feature_names_in_)

    return transformed_data

# API 엔드포인트 생성
@app.post("/predict_delay", response_model=PredictionOutput)
def predict_flight_status(input_data: FlightInput):
    transformed_input = transform_user_input(input_data)
    probabilities = model.predict_proba(transformed_input)[0]
    status_mapping = {0: "departure", 1: "delay", 2: "cancellation", 3: "return"}
    
    # ✅ 동적으로 확률 값 반환 (예제가 아니라, 실제 값!)
    result = {status_mapping[i]: round(probabilities[i] * 100, 2) for i in range(len(probabilities))}
    
    return {"prediction_probabilities": result}

###########################################################################################################################################################
## 실시간 인천공항 현황 확인 ## 
###########################################################################################################################################################

cached_forecast = None  # 🔄 캐싱된 출국장 혼잡도 데이터 저장 변수

# ✅ 출국장 혼잡도 기준
def determine_congestion_level(passenger_count, previous_levels):
    if passenger_count >= 8600 or (passenger_count > 8200 and previous_levels[-2:] == ["ORANGE", "ORANGE"]):
        return "RED"
    elif passenger_count > 8200:
        return "ORANGE"
    elif passenger_count > 7600:
        return "YELLOW"
    else:
        return "BLUE"

def fetch_departure_forecast():
    url = 'http://apis.data.go.kr/B551177/PassengerNoticeKR/getfPassengerNoticeIKR'
    params = {
        'serviceKey' : SERVICE_KEY,
        'selectdate': "0",
        'type': 'json'
    }

    response = requests.get(url, params=params)

    try:
        data = response.json()

        if 'response' in data:

            if 'body' in data['response'] and 'items' in data['response']['body']:
                print(f"✅ API 응답 데이터 정상 수신!")

                items = data['response']['body']['items']
                if not items:
                    print("⚠ 데이터가 비어 있음!")
                    return {"error": "No data available from API"}

                result = []
                prev_levels = []

                for item in items:
                    if item["atime"] != "합계":
                        t1_departure_1_2 = float(item.get("t1sum5", 0))
                        t1_departure_3 = float(item.get("t1sum6", 0))
                        t1_departure_4 = float(item.get("t1sum7", 0))
                        t1_departure_5_6 = float(item.get("t1sum8", 0))
                        t1_total = float(item.get("t1sumset2", 0))
                        t2_departure_1 = float(item.get("t2sum3", 0))
                        t2_departure_2 = float(item.get("t2sum4", 0))
                        t2_total = float(item.get("t2sumset2", 0))
                        total_departure = t1_total + t2_total
                        congestion_level = determine_congestion_level(total_departure, prev_levels)
                        prev_levels.append(congestion_level)

                        result.append({
                            "time_range": item["atime"],
                            "T1_departure_1_2": t1_departure_1_2,
                            "T1_departure_3": t1_departure_3,
                            "T1_departure_4": t1_departure_4,
                            "T1_departure_5_6": t1_departure_5_6,
                            "T1_total": t1_total,
                            "T2_departure_1": t2_departure_1,
                            "T2_departure_2": t2_departure_2,
                            "T2_total": t2_total,
                            "total_departure": total_departure,
                            "congestion_level": congestion_level
                        })
                return {"departure_forecast": result}

        return {"error": "Invalid API response format"}

    except Exception as e:
        print(f"🚨 API 호출 중 예외 발생: {e}")

    return {"error": "Failed to fetch data"}




@app.on_event("startup")
@repeat_every(seconds=3600)  # 🔄 1시간마다 업데이트
def update_departure_forecast():
    global cached_forecast
    cached_forecast = fetch_departure_forecast()
    print("🔄 출국장 혼잡도 데이터 업데이트 완료")  

# ✅ 응답 데이터 모델
class DepartureData(BaseModel):
    time_range: str = Field(..., example="07_08")
    T1_departure_1_2: float = Field(..., example=500.0)
    T1_departure_3: float = Field(..., example=200.0)
    T1_departure_4: float = Field(..., example=300.0)
    T1_departure_5_6: float = Field(..., example=400.0)
    T1_total: float = Field(..., example=1400.0)
    T2_departure_1: float = Field(..., example=300.0)
    T2_departure_2: float = Field(..., example=400.0)
    T2_total: float = Field(..., example=700.0)
    total_departure: float = Field(..., example=2100.0)
    congestion_level: str = Field(..., example="ORANGE")

class DepartureForecastResponse(BaseModel):
    departure_forecast: List[DepartureData]

@app.get("/passenger_status", response_model=DepartureForecastResponse)
def get_cached_departure_forecast():
    
    if cached_forecast and "departure_forecast" in cached_forecast:
        return cached_forecast
    
    return {"error": "No cached data available"}

###########################################################################################################################################################
## 실시간 주차 정보 ##
###########################################################################################################################################################

# 🔄 캐싱된 주차 데이터 저장
cached_parking_status = None

# ✅ 공공데이터 API에서 주차 데이터 가져오기
def fetch_parking_status():
    url = "http://apis.data.go.kr/B551177/StatusOfParking/getTrackingParking"
    params = {
        "serviceKey": SERVICE_KEY,
        "numOfRows": "10000",
        "pageNo": "1",
        "type": "json"
    }

    response = requests.get(url, params=params)

    try:
        data = response.json()
        if "response" in data and "body" in data["response"] and "items" in data["response"]["body"]:
            items = data["response"]["body"]["items"]

            # T1과 T2 데이터를 따로 저장
            parking_data = {"T1": [], "T2": []}

            for item in items:
                floor_name = item["floor"]  # 🔥 UTF-8 그대로 사용!
                parking_total = int(item["parkingarea"])
                parking_used = int(item["parking"])
                parking_available = parking_total - parking_used
                occupancy_rate = round((parking_used / parking_total) * 100, 2) if parking_total > 0 else 0

                # JSON 형태 유지하면서 추가 정보 포함
                parking_info = {
                    "floor": floor_name,
                    "parking": parking_used,  # 현재 주차된 차량 수
                    "parkingarea": parking_total,  # 총 주차 공간 수
                    "available_spots": parking_available,  # 사용 가능한 주차 공간 수
                    "occupancy_rate": f"{occupancy_rate}%",  # 사용률(%)
                    "datetm": item["datetm"]  # 데이터 업데이트 시간
                }

                # T1 / T2 분류
                if "T1" in floor_name:
                    parking_data["T1"].append(parking_info)
                elif "T2" in floor_name:
                    parking_data["T2"].append(parking_info)

            return {"parking_status": parking_data}

    except Exception as e:
        print(f"🚨 API 호출 중 오류 발생: {e}")

    return {"error": "Failed to fetch data"}

# 🔄 1시간마다 자동 업데이트
@app.on_event("startup")
@repeat_every(seconds=3600)
def update_parking_status():
    global cached_parking_status
    cached_parking_status = fetch_parking_status()
    print("🔄 실시간 주차장 데이터 업데이트 완료")

# ✅ 응답 데이터 모델
class ParkingInfo(BaseModel):
    floor: str = Field(..., example="T1 단기주차장")
    parking: int = Field(..., example=1000)  # 현재 주차된 차량 수
    parkingarea: int = Field(..., example=1200)  # 총 주차 공간
    available_spots: int = Field(..., example=200)  # 사용 가능한 공간
    occupancy_rate: str = Field(..., example="83.3%")  # 사용률
    datetm: str = Field(..., example="20250216162242.703")  # 데이터 업데이트 시간

class ParkingResponse(BaseModel):
    parking_status: dict[str, list[ParkingInfo]]

# ✅ GET 요청: 캐싱된 데이터 제공
@app.get("/parking_status", response_model=ParkingResponse)
def get_parking_status():
    if cached_parking_status and "parking_status" in cached_parking_status:
        return cached_parking_status
    return {"error": "No cached data available"}

###########################################################################################################################################################
## 타 항공사 승객 예고 현황
###########################################################################################################################################################

cached_passenger_data = None  # ✅ 캐싱된 데이터 저장 변수

# 📌 특정 공항의 데이터 가져오기 (공항 코드: GMP, CJJ, CJU)
def fetch_passenger_data_for_airport(airport_code):
    today = datetime.today().strftime('%Y%m%d')  # 🔹 오늘 날짜 자동 설정
    
    url = 'http://openapi.airport.co.kr/service/rest/dailyExpectPassenger/dailyExpectPassenger'
    params = {
        'serviceKey': SERVICE_KEY_2,
        'schDate': today,
        'schAirport': airport_code  # ✅ 특정 공항 지정
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None

    return parse_passenger_data(response.content, airport_code)

# 📌 XML 데이터 파싱 (공항별 데이터 변환)
def parse_passenger_data(xml_data, airport_code):
    root = ET.fromstring(xml_data)
    airport_data = {airport_code: {"international": {}, "domestic": {}}}

    for item in root.findall(".//item"):
        time_slot = item.findtext("hh", "").strip()  # 시간대 (00~23시)
        passenger_count = int(item.findtext("pct", 0))  # 예상 승객 수
        group_passenger = int(item.findtext("pcg", 0))  # 단체 승객 수
        congestion = item.findtext("congestYn", "N").strip()  # 혼잡 여부
        flight_type = "international" if item.findtext("tof", "N").strip() == "I" else "domestic"  # 국내/국제 구분

        airport_data[airport_code][flight_type][time_slot] = {
            "passenger_count": passenger_count,
            "group_passenger": group_passenger,
            "congestion": congestion
        }

    return airport_data

# 📌 모든 공항 데이터 통합 호출
def fetch_all_passenger_data():
    airports = ["GMP", "CJJ", "CJU"]  # ✅ 김포, 청주, 제주 공항 조회
    merged_data = {"airport_data": {}}

    for airport in airports:
        airport_data = fetch_passenger_data_for_airport(airport)
        if airport_data:
            merged_data["airport_data"].update(airport_data)

    return merged_data

# 🔄 1시간마다 자동으로 데이터 업데이트
@app.on_event("startup")
@repeat_every(seconds=3600)
def update_passenger_data():
    global cached_passenger_data
    cached_passenger_data = fetch_all_passenger_data()
    print("🔄 김포, 청주, 제주 공항 승객 수 데이터 업데이트 완료")

# ✅ 응답 데이터 모델 정의
class TimeSlotData(BaseModel):
    passenger_count: int = Field(..., example=300)
    group_passenger: int = Field(..., example=50)
    congestion: str = Field(..., example="N")

class FlightTypeData(BaseModel):
    international: Dict[str, TimeSlotData] = Field(
        default_factory=dict,
        example={
            "00": {"passenger_count": 175, "group_passenger": 0, "congestion": "N"},
            "05": {"passenger_count": 142, "group_passenger": 0, "congestion": "N"},
        }
    )
    domestic: Dict[str, TimeSlotData] = Field(
        default_factory=dict,
        example={
            "07": {"passenger_count": 160, "group_passenger": 0, "congestion": "N"},
            "08": {"passenger_count": 88, "group_passenger": 0, "congestion": "N"},
        }
    )

class AirportData(BaseModel):
    airport_data: Dict[str, FlightTypeData] = Field(
        default_factory=dict,
        example={
            "GMP": {
                "international": {
                    "06": {"passenger_count": 500, "group_passenger": 30, "congestion": "Y"},
                    "07": {"passenger_count": 620, "group_passenger": 40, "congestion": "Y"},
                },
                "domestic": {
                    "08": {"passenger_count": 250, "group_passenger": 20, "congestion": "N"},
                    "09": {"passenger_count": 340, "group_passenger": 25, "congestion": "N"},
                },
            },
            "CJU": {
                "international": {
                    "06": {"passenger_count": 400, "group_passenger": 20, "congestion": "N"},
                },
                "domestic": {
                    "07": {"passenger_count": 150, "group_passenger": 10, "congestion": "N"},
                },
            },
        }
    )

# ✅ response_model을 지정하여 JSON 응답을 정확하게 문서화
@app.get("/other_status", response_model=AirportData)
def get_passenger_data():
    if cached_passenger_data:
        return cached_passenger_data
    return {"error": "No cached data available"}

###########################################################################################################################################################
## 공항 기상 정보
###########################################################################################################################################################

# 응답 데이터 모델 정의
class WeatherInfo(BaseModel):
    date: str
    title: str
    summary: str
    outlook: str
    forecast: str
    warn: str
    sel_val1: str
    sel_val2: str
    sel_val3: str

@app.get("/get_weather", response_model=WeatherInfo)
def get_incheon_weather():
    # 현재 시간 확인
    current_time = datetime.now()
    current_hour = current_time.hour

    # base_time 결정 (17:00 지나면 1700, 아니면 0600)
    base_time = '1700' if current_hour >= 17 else '0600'

    # API 요청 URL 및 파라미터 설정
    url = 'http://apis.data.go.kr/1360000/AirPortService/getAirPort'
    params = {
        'serviceKey': SERVICE_KEY,
        'numOfRows': '1000',
        'pageNo': '1',
        'dataType': 'JSON',
        'base_date': current_time.strftime('%Y%m%d'),  # 오늘 날짜
        'base_time': base_time,  # 결정된 base_time
        'airPortCd': 'RKSI'  # 인천공항 코드
    }

    # API 요청 및 응답 처리
    response = requests.get(url, params=params)
    data = response.json()

    # 응답 데이터 처리
    if 'response' in data and 'body' in data['response']:
        weather_info = data['response']['body']['items']['item'][0]

        # 각 항목의 값이 비어있다면 적절히 처리
        return WeatherInfo(
            date=weather_info.get('tm', '정보 없음'),
            title=weather_info.get('title', '정보 없음'),
            summary=weather_info.get('summary', '정보 없음'),
            outlook=weather_info.get('outlook', '정보 없음'),
            forecast=weather_info.get('forecast', '정보 없음'),
            warn=weather_info.get('warn', '정보 없음'),
            sel_val1=weather_info.get('sel_val1', '정보 없음'),
            sel_val2=weather_info.get('sel_val2', '정보 없음'),
            sel_val3=weather_info.get('sel_val3', '정보 없음')
        )
    else:
        return {"error": "데이터를 가져오는 데 실패했습니다."}
