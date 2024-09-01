import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import argparse
from tqdm.auto import tqdm

# 명령줄 인수를 처리하기 위해 argparse 사용
parser = argparse.ArgumentParser(description='크롤링할 연도 입력')
parser.add_argument('Year', type=int, help='크롤링할 연도 (예: 2016)')
args = parser.parse_args()

Year = args.Year
base_url = f'https://www.kboat.or.kr/contents/information/raceResultList.do?stndYear={Year}'

# HTTP GET 요청
response = requests.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')

# 회차 추출
rounds = [option['value'] for option in soup.select('select[name="tms"] option')]

# 각 회차별 일차 수를 확인하는 과정
round_day_pairs = []

for Time in tqdm(rounds):
    # 회차 1일차 페이지로 이동
    url = f'{base_url}&tms={Time}&dayOrd=1'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 일차 옵션 추출
    days = [option['value'] for option in soup.select('select[name="dayOrd"] option')]

    # (회차, 일차) 조합 추가
    for Day in days:
        round_day_pairs.append((Time, Day))

print(f'{Year}연도, 총 {len(round_day_pairs)}개의 (회차, 일차) Pair 수집 완료')

# 결과 데이터를 담을 리스트
all_data = []

# 추출된 (회차, 일차) 조합을 바탕으로 데이터 크롤링
for Time, Day in tqdm(round_day_pairs):
    # URL 정의
    url = f'{base_url}&tms={Time}&dayOrd={Day}'

    # HTTP GET 요청
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 먼저 <div class="table pcType">를 찾고, 그 안에서 <tbody>와 <tr>을 찾기
    table_div = soup.select_one('div.table.pcType')
    if table_div:
        rows = table_div.select('tbody tr')

        # 각 행에서 데이터 추출
        for row in rows:
            cols = row.find_all('td')

            # 기본값 설정 (필요한 열이 없을 경우 대비)
            race_no = cols[0].text.strip() if len(cols) > 0 else pd.NA
            first_place = re.sub(r'\s+', ' ', cols[1].text.strip()) if len(cols) > 1 else pd.NA
            second_place = re.sub(r'\s+', ' ', cols[2].text.strip()) if len(cols) > 2 else pd.NA
            third_place = re.sub(r'\s+', ' ', cols[3].text.strip()) if len(cols) > 3 else pd.NA
            dansung = cols[4].text.strip() if len(cols) > 4 else pd.NA
            yeonsung = re.sub(r'\s*\n\s*', ', ', cols[5].text.strip()) if len(cols) > 5 else pd.NA
            ssangsung = cols[6].text.strip() if len(cols) > 6 else pd.NA
            boksung = cols[7].text.strip() if len(cols) > 7 else pd.NA
            samboksung = cols[8].text.strip() if len(cols) > 8 else pd.NA
            쌍복승 = cols[9].text.strip() if len(cols) > 9 else pd.NA
            삼쌍승 = cols[10].text.strip() if len(cols) > 10 else pd.NA
            Refund = cols[11].text.strip() if len(cols) > 11 else pd.NA

            # 데이터를 리스트에 추가
            all_data.append([
                Year, Time, Day, race_no,
                first_place, second_place, third_place,
                dansung, yeonsung, ssangsung, boksung, samboksung, 쌍복승, 삼쌍승, Refund])


columns = ['연도', '회차', '일차', '경주', '1위', '2위', '3위', '단승식', '연승식', '쌍승식', '복승식', '삼복승식', '쌍복승', '삼쌍승', 'Refund']
df = pd.DataFrame(all_data, columns=columns)

# 공백 또는 공백만 포함된 값을 NaN으로 대체
df.replace(r'^\s*$', pd.NA, inplace=True, regex=True)
df.dropna(thresh=5, inplace=True)  # 결측치가 threshold 이상인 행 삭제

# CSV 파일로 저장
output_file = f'./crawlled_data/kboat_result_{Year}.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f'{Year} 경기 결과 수집 완료, 파일 저장: {output_file}')
