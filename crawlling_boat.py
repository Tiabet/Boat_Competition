import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import argparse
from tqdm.auto import tqdm

def crawl_motor_info(year):
    base_url = f'https://www.kboat.or.kr/contents/information/raceRecordList.do?repairCode=B&searchYear={year}'

    # HTTP GET 요청
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 페이지 번호 추출
    pagination = soup.select('ul.pagination li a')
    page_nums = [int(re.search(r'cPage=(\d+)', link['href']).group(1)) for link in pagination if 'cPage' in link['href']]
    max_page_num = max(page_nums) if page_nums else 1

    # 모든 페이지에서 데이터를 크롤링할 리스트
    all_motor_data = []

    # 각 페이지에서 데이터 크롤링
    for page_num in tqdm(range(1, max_page_num + 1)):
        url = f'{base_url}&cPage={page_num}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # <table class="tb_data4 mb30"> 내에서 <tbody>와 <tr> 찾기
        table = soup.select_one('table.tb_data4.mb30')
        rows = table.select('tbody tr') if table else []

        # 각 행에서 데이터 추출
        for row in rows:
            cols = row.find_all('td')

            Boat_No = cols[0].text.strip()
            Year = cols[1].text.strip().split('/')[0].strip()
            출전회차 = cols[1].text.strip().split('/')[1].strip()
            출주회수 = cols[2].text.strip()
            일착 = cols[3].text.strip()
            이착 = cols[4].text.strip()
            삼착 = cols[5].text.strip()
            평균착순점 = cols[6].text.strip()
            연대율 = cols[7].text.strip()
            삼주회 = cols[8].text.strip()
            이주회_온라인 = cols[9].text.strip() if len(cols) > 9 else ''
            이주회_플라잉 = cols[10].text.strip() if len(cols) > 10 else ''

            # 데이터를 리스트에 추가
            all_motor_data.append([Year, Boat_No, 출전회차, 출주회수, 일착, 이착, 삼착, 평균착순점, 연대율, 삼주회, 이주회_온라인, 이주회_플라잉])

    # Pandas DataFrame으로 변환
    columns = ['Year', 'Boat_No', '출전회차', '출주회수', '1착', '2착', '3착', '평균착순점', '연대율', '3주회', '2주회(온라인)', '2주회(플라잉)']
    df = pd.DataFrame(all_motor_data, columns=columns)

    # CSV 파일로 저장
    df.to_csv(f'./crawlled_data/kboat_boat_{year}.csv', index=False, encoding='utf-8-sig')
    print(f'{year}년도 모터 정보 크롤링 완료, kboat_boat_{year}.csv로 저장되었습니다.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='크롤링할 연도를 입력하세요.')
    parser.add_argument('Year', type=int, help='크롤링할 연도 (예: 2016)')
    args = parser.parse_args()

    crawl_motor_info(args.Year)
