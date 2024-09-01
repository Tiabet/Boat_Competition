import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
from tqdm import tqdm

def get_tms_and_dayOrd(year):
    회차=4
    일차=1

    # URL 설정
    url = f'https://www.kboat.or.kr/contents/information/fixedChuljuPage.do?stndYear={year}&tms={회차}&dayOrd={일차}'

    # HTTP GET 요청
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 회차(tms)와 일차(dayOrd) 값 추출
    tms_options = soup.select('select[name="tms"] option')
    dayOrd_options = soup.select('select[name="dayOrd"] option')

    tms_list = [int(option['value']) for option in tms_options]
    dayOrd_list = [int(option['value']) for option in dayOrd_options]

    return tms_list, dayOrd_list

def crawl_race_entries(year, tms, day_ord):
    base_url = f'https://www.kboat.or.kr/contents/information/fixedChuljuPage.do?stndYear={year}&tms={tms}&dayOrd={day_ord}&race_no='

    # HTTP GET 요청
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 경주 수 확인
    race_headers = soup.select('h4.titPlayChart')
    total_races = len(race_headers) // 2

    # 전체 데이터를 저장할 리스트 초기화
    all_data = []

    # 경주 번호에 따라 반복하여 데이터 수집
    race_blocks = soup.select('.pcType')  # 각 경주 블록들을 선택

    for race_no in range(1, total_races + 1):
        try:
            # 해당 경주에 대한 블록 가져오기
            race_block = race_blocks[race_no - 1]

            # 첫 번째 테이블 데이터 추출
            table1 = race_block.select_one('.table.bd table')
            rows1 = table1.select('tr') if table1 else []

            table2 = race_block.select_one('.table.bd.mb50 table')
            rows2 = table2.select('tbody tr') if table2 else []

            data_1 = []
            data_2 = []

            for row in rows1[3:]:
                cols = row.find_all('td')

                # 열의 개수가 예상보다 적으면 해당 경주 건너뛰기
                번호 = cols[0].text.strip()
                등급 = cols[1].text.strip()
                기수 = cols[2].text.strip()
                선수명 = cols[3].text.strip()
                성별 = cols[4].text.strip()
                나이 = cols[5].text.strip()
                체중 = cols[6].text.strip()
                최근6회차_평균착순점 = cols[7].text.strip()
                최근6회차_평균득점 = cols[8].text.strip()
                최근6회차_승률 = cols[9].text.strip()
                최근6회차_연대율2 = cols[10].text.strip()
                최근6회차_연대율3 = cols[11].text.strip()
                최근6회차_평균ST = cols[12].text.strip()
                최근8경주_착순 = cols[13].text.strip()
                연간성적_평균착순점 = cols[14].text.strip()
                연간성적_연대율 = cols[15].text.strip()
                FL = cols[16].text.strip()
                평균사고점 = cols[17].text.strip()
                금일출주경주 = cols[18].text.strip()
                전일성적 = cols[19].text.strip()

                # 데이터를 리스트에 추가
                data_1.append([
                    year, tms, day_ord, race_no, 번호, 등급, 기수, 선수명, 성별, 나이, 체중,
                    최근6회차_평균착순점, 최근6회차_평균득점, 최근6회차_승률, 최근6회차_연대율2,
                    최근6회차_연대율3, 최근6회차_평균ST, 최근8경주_착순, 연간성적_평균착순점,
                    연간성적_연대율, FL, 평균사고점, 금일출주경주, 전일성적
                ])

            columns_1 = [
                '연도', '회차', '일차', '경주번호', '번호', '등급', '기수', '선수명', '성별', '나이', '체중',
                '최근6회차_평균착순점', '최근6회차_평균득점', '최근6회차_승률', '최근6회차_연대율2',
                '최근6회차_연대율3', '최근6회차_평균ST', '최근8경주_착순', '연간성적_평균착순점',
                '연간성적_연대율', 'FL', '평균사고점', '금일출주경주', '전일성적'
            ]
            df_1 = pd.DataFrame(data_1, columns=columns_1)

            for row2 in rows2:
                cols2 = row2.find_all('td')
                출주횟수 = cols2[2].text.strip()
                코스_1코스 = cols2[3].text.strip()
                코스_2코스 = cols2[4].text.strip()
                코스_3코스 = cols2[5].text.strip()
                코스_4코스 = cols2[6].text.strip()
                코스_5코스 = cols2[7].text.strip()
                코스_6코스 = cols2[8].text.strip()
                모터번호 = cols2[9].text.strip()
                모터_평균착순점 = cols2[10].text.strip()
                모터_연대율2 = cols2[11].text.strip()
                모터_연대율3 = cols2[12].text.strip()
                전탑승선수1 = cols2[13].text.strip()
                전탑승선수2 = cols2[14].text.strip()
                보트번호 = cols2[15].text.strip()
                보트_평균착순점 = cols2[16].text.strip()
                보트_연대율 = cols2[17].text.strip()
                특이사항 = cols2[18].text.strip() if len(cols2) > 18 else ''

                # 데이터를 리스트에 추가
                data_2.append([
                    출주횟수,
                    코스_1코스, 코스_2코스, 코스_3코스, 코스_4코스, 코스_5코스, 코스_6코스,
                    모터번호, 모터_평균착순점, 모터_연대율2, 모터_연대율3, 전탑승선수1,
                    전탑승선수2, 보트번호, 보트_평균착순점, 보트_연대율, 특이사항
                ])

            # Pandas DataFrame으로 변환
            columns_2 = [
                '출주횟수',
                '코스_1코스', '코스_2코스', '코스_3코스', '코스_4코스', '코스_5코스', '코스_6코스',
                '모터번호', '모터_평균착순점', '모터_연대율2', '모터_연대율3', '전탑승선수1',
                '전탑승선수2', '보트번호', '보트_평균착순점', '보트_연대율', '특이사항'
            ]
            df_2 = pd.DataFrame(data_2, columns=columns_2)

            # 두 DataFrame을 하나로 합치기
            df = pd.concat([df_1, df_2], axis=1)
            all_data.append(df)

        except IndexError as e:
            print(f"Skipping race: {year} {tms} {day_ord} {race_no} due to error: {e}")
            continue

    # 전체 데이터를 하나의 DataFrame으로 반환
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python crawlling_entry.py <year>")
        sys.exit(1)

    year = int(sys.argv[1])
    tms_list, dayOrd_list = get_tms_and_dayOrd(year)
    tms_list = sorted(tms_list)
    dayOrd_list = sorted(dayOrd_list)
    print(f'({tms_list[0]}회 {dayOrd_list[0]}일차 부터 {tms_list[-1]}회 {dayOrd_list[-1]}일차까지, 총 {len(tms_list)*len(dayOrd_list)}개 수집 시작')

    all_entries_data = []

    j = 1
    for tms in tqdm(tms_list, leave=False):
        for day_ord in dayOrd_list:
            df = crawl_race_entries(year, tms, day_ord)
            all_entries_data.append(df)
            if j % 10 == 0:
                print(f"{tms}회차 완료")

            j += 1

    # 전체 데이터를 하나의 DataFrame으로 합치기
    final_df = pd.concat(all_entries_data, ignore_index=True)

    # CSV 파일로 저장
    final_df.to_csv(f'./crawlled_data/kboat_entries_{year}.csv', index=False, encoding='utf-8-sig')
    print(f'{year}년 모든 출주표 정보 크롤링 완료')


### ex) python crawlling_entry.py 2016