import pandas as pd

# ------------------------------
# 공통 함수
# ------------------------------
def process_facility_df(df, column_name, years=None):
    """
    특정 컬럼과 연도(years)를 기준으로 학군별 합계를 구하는 함수
    
    Parameters:
        df (pd.DataFrame): 원본 데이터
        column_name (str): 합계를 낼 대상 컬럼
        years (list, optional): 사용할 연도 리스트 (문자열)
        
    Returns:
        pd.DataFrame: 학군별 합계 pivot
    """
    if years is None:
        years = [str(y) for y in range(2018, 2024)]

    grouped = df.groupby(["학군", "년도"])[column_name].sum().reset_index()
    pivot = grouped.pivot(index="학군", columns="년도", values=column_name).fillna(0)

    pivot = pivot[years]
    pivot.columns = [f"{column_name}_{y}" for y in years]
    return pivot


# ------------------------------
# 데이터 처리
# ------------------------------

# 예산 데이터 (2018~2023)
budget_years = [str(y) for y in range(2018, 2024)]
budget_grouped = process_facility_df(budget, "교육예산(합)", budget_years)

# 청소년시설, 지역아동시설, 공공도서관
youth_grouped = process_facility_df(youth, "청소년시설")
child_grouped = process_facility_df(child, "지역아동시설")
library_grouped = process_facility_df(library, "공공도서관")

# ------------------------------
# 데이터 병합
# ------------------------------
final_df = budget_grouped.join(
    [youth_grouped, child_grouped, library_grouped], how="outer"
)

# 학군 순서 정렬
region_order = ['동부', '서부', '남부', '북부', '중부',
                '강동송파', '강서', '강남', '동작관악', '성동광장', '성북']

final_df = final_df.reset_index()
final_df['학군'] = pd.Categorical(final_df['학군'], categories=region_order, ordered=True)
final_df = final_df.sort_values('학군').set_index('학군')
