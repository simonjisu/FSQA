import time
import yaml
from tqdm.notebook import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd
import OpenDartReader
import FinanceDataReader as fdr

import psycopg
from psycopg import sql
from pathlib import Path
from tqdm import tqdm


def create_database(db_settings):
    with psycopg.connect(user=db_settings['user'], password=db_settings['password'], autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT EXISTS( SELECT 1 FROM pg_database WHERE datname='{db_settings['dbname']}');")
            exists = cur.fetchone()[0]
        if not exists:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(db_settings['dbname'])))
        else:
            print(f"Existence of Database {db_settings['dbname']}: {exists}")

def create_table(db_settings, table_names):
    # if the database exists: create tables
    commands = (
        """
        CREATE TABLE company (
            c_id SERIAL PRIMARY KEY,
            code VARCHAR(10) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            sector VARCHAR(255) NOT NULL
        )
        """,
        """ 
        CREATE TABLE financial_statement (
            fs_id SERIAL PRIMARY KEY,
            code VARCHAR(10) NOT NULL,
            rpid VARCHAR(30) UNIQUE NOT NULL,
            bsns_year INT NOT NULL,
            FOREIGN KEY (code) REFERENCES company (code) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE balance_sheet (
            bs_id SERIAL PRIMARY KEY,
            code VARCHAR(10) NOT NULL,
            rpid VARCHAR(30) NOT NULL,
            quarter VARCHAR(10) NOT NULL,
            account VARCHAR(255) NOT NULL,
            value BIGINT NOT NULL,
            FOREIGN KEY (code) REFERENCES company (code) ON DELETE CASCADE,
            FOREIGN KEY (rpid) REFERENCES financial_statement (rpid) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE income_statement (
            is_id SERIAL PRIMARY KEY,
            code VARCHAR(10) NOT NULL,
            rpid VARCHAR(30) NOT NULL,
            quarter VARCHAR(10) NOT NULL,
            account VARCHAR(255) NOT NULL,
            value BIGINT NOT NULL,
            FOREIGN KEY (code) REFERENCES company (code) ON DELETE CASCADE,
            FOREIGN KEY (rpid) REFERENCES financial_statement (rpid) ON DELETE CASCADE
        )
        """
    )

    conn = None
    with psycopg.connect(
            dbname=db_settings['dbname'],
            host=db_settings['host'],
            user=db_settings['user'], 
            password=db_settings['password'],
            port=db_settings['port']
        ) as conn:
        with conn.cursor() as cur:
            for i, command in enumerate(commands):
                cur.execute(
                    f"SELECT EXISTS( SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_names[i]}');"
                )
                exists = cur.fetchone()[0]
                if not exists:     
                    cur.execute(sql.SQL(command))
                    print(f'Table created: {table_names[i]}')
                else:
                    print(f'Table already exists: {table_names[i]}')
        conn.commit()
        
def insert_company(db_settings, stocks, tables, corp):
    with psycopg.connect(
        dbname=db_settings['dbname'],
        host=db_settings['host'],
        user=db_settings['user'], 
        password=db_settings['password'],
        port=db_settings['port']
    ) as conn:
        with conn.cursor() as cur:
            for i, line in stocks.loc[stocks['Symbol']==corp, ['Symbol', 'Name', 'Sector']].iterrows():
                cur.execute(
                    sql.SQL("INSERT INTO {}({}, {}, {}, {}) VALUES (DEFAULT, %s, %s, %s) ;").format(
                        sql.Identifier('company'), *list(map(sql.Identifier, tables['company']))
                    ),
                    tuple(line.values)
                )

def convert_to_q(report_code, bsns_year, bsns_year_dict):
    report_code_word_dict = {
        '11011': ('', '4Q'), 
        '11013': (' 1분기', '1Q'), 
        '11012': (' 반기', '2Q'), 
        '11014': (' 3분기', '3Q')
    }

    s = f'제 {bsns_year_dict[bsns_year]} 기'
    w, q = report_code_word_dict[report_code]
    if report_code == '11011':
        return {s+w:q}
    else:
        return {s+w+'말':q, s+w:q}


if __name__ == '__main__':
    # Load settings
    main_path = Path('.').absolute().parent
    with (main_path / 'setting_files' / 'app_settings.yml').open('r') as file:
        settings = yaml.full_load(file)
    
    # pleas add your custom operdart key from https://opendart.fss.or.kr/intro/main.do
    dart_settings = settings['opendart']  
    dart = OpenDartReader(dart_settings['apikey'])

    stocks = fdr.StockListing('KOSPI')
    stocks = stocks.loc[~stocks['Sector'].isnull(), :]
    stocks_syms = stocks['Symbol'].values
    stocks = stocks.reset_index(drop=True)
    stocks.fillna('', inplace=True)
    for c in stocks.columns:
        print(c, stocks[c].dtype)

    db_settings = settings['db']
    create_database(db_settings)
    table_names = ['company', 'financial_statement', 'balance_sheet', 'income_statement', 'cashflow_statement']
    create_table(db_settings, table_names)
    
    table_columns = [
        ('c_id', 'code', 'name', 'sector'),
        ('fs_id', 'code', 'rpid', 'bsns_year'),
        ('bs_id', 'code', 'rpid', 'quarter', 'account', 'value'),
        ('is_id', 'code', 'rpid', 'quarter', 'account', 'value'),
    ]
    tables = dict(zip(table_names, table_columns))

    corp = '005930'
    insert_company(db_settings, stocks, tables, corp)

    data_path = main_path / 'src' / 'data'
    df_account = pd.read_csv(data_path / 'AccountName.csv')
    account_eng_dict = dict(zip(df_account['acc_name_kor'].values, df_account['acc'].values))
    bsns_years = list(range(2015, 2022))
    report_codes = ['11011', '11013', '11012', '11014']
    cols = ['rcept_no', 'thstrm_nm', 'account_nm', 'thstrm_amount']
    bsns_year_dict = dict(zip(bsns_years, range(47, 47+len(bsns_years))))
    accounts_dict = {
        'balance_sheet' : [
            '유동자산', '현금및현금성자산', '매출채권', '선급비용', '재고자산', '비유동자산', '유형자산', '무형자산', '자산총계',
            '유동부채', '매입채무', '단기차입금', '선수금', '비유동부채', '사채', '장기차입금', '부채총계', '자본총계', '부채와자본총계'
        ],
        'income_statement': [
            '수익(매출액)', '매출원가', '매출총이익', '판매비와관리비', '영업이익', '금융수익', '금융비용',
            '법인세비용차감전순이익(손실)', '법인세비용', '계속영업이익(손실)', '당기순이익(손실)'
        ]
    }

    data = defaultdict(list)
    for bsns_year in bsns_years:
        for report_code in report_codes:
            df = dart.finstate_all(corp, bsns_year, reprt_code=report_code, fs_div='CFS')
            time.sleep(0.5)
            if df is None:
                continue
            
            # rcept_no
            recpt_no = df['rcept_no'].values[0]
            q_dict = convert_to_q(report_code, bsns_year, bsns_year_dict)
            # bs
            sj = 'BS'
            df_bs = df.loc[(df['sj_div'] == sj), cols]    #  & df['account_nm'].isin(accounts_dict['balance_sheet'])     
            df_bs.loc[:, 'thstrm_nm'] = df_bs['thstrm_nm'].apply(q_dict.get)
            # is
            sj = 'CIS' if df['sj_div'].isin(['IS']).sum() == 0 else 'IS'
            df_is = df.loc[(df['sj_div'] == sj), cols] # & df['account_nm'].isin(accounts_dict['income_statement'])
            df_is.loc[:, 'thstrm_nm'] = df_is['thstrm_nm'].apply(q_dict.get)
            data['info'].append([corp, recpt_no, f'{bsns_year}'])
            data['bs'].append(df_bs)
            data['is'].append(df_is)

    # save
    data['info'] = pd.DataFrame(data['info'], columns=['code', 'rpid', 'time'])
    data['BS'] = pd.concat(data['bs']).reset_index(drop=True)
    data['IS'] = pd.concat(data['is']).reset_index(drop=True)

    data['BS'].loc[:, 'account_nm'] = data['BS']['account_nm'].apply(account_eng_dict.get)
    data['IS'].loc[:, 'account_nm'] = data['IS']['account_nm'].apply(account_eng_dict.get)

    data['BS'] = data['BS'].loc[~data['BS'].loc[:, 'account_nm'].isnull(), :].reset_index(drop=True)
    data['IS'] = data['IS'].loc[~data['IS'].loc[:, 'account_nm'].isnull(), :].reset_index(drop=True)

    with psycopg.connect(
        dbname=db_settings['dbname'],
        host=db_settings['host'],
        user=db_settings['user'], 
        password=db_settings['password'],
        port=db_settings['port']
    ) as conn:
        with conn.cursor() as cur:
            for i, line in data['info'].iterrows():
                cur.execute(
                    sql.SQL("INSERT INTO {}({}, {}, {}, {}) VALUES (DEFAULT, %s, %s, %s) ;").format(
                        sql.Identifier('financial_statement'), *list(map(sql.Identifier, tables['financial_statement']))
                    ),
                    tuple(line.values)
                )

            for i, line in data['BS'].iterrows():
                cur.execute(
                    sql.SQL("INSERT INTO {}({}, {}, {}, {}, {}, {}) VALUES (DEFAULT, %s, %s, %s, %s, %s) ;").format(
                        sql.Identifier('balance_sheet'), *list(map(sql.Identifier, tables['balance_sheet']))
                    ),
                    tuple([corp] + list(line.values))
                )

            for i, line in data['IS'].iterrows():
                cur.execute(
                    sql.SQL("INSERT INTO {}({}, {}, {}, {}, {}, {}) VALUES (DEFAULT, %s, %s, %s, %s, %s) ;").format(
                        sql.Identifier('income_statement'), *list(map(sql.Identifier, tables['income_statement']))
                    ),
                    tuple([corp] + list(line.values))
                )

            # CREATE VIEW TABLE
            # VIEW TABLE: Income Statement
            cur.execute(
            """
            CREATE VIEW vt_is_005930
            AS 
            SELECT fs.bsns_year AS bsns_year, inc.quarter AS quarter, inc.account AS account, inc.value AS value
            FROM income_statement AS inc
            INNER JOIN company AS com
                ON com.code = inc.code
            INNER JOIN financial_statement AS fs
                ON fs.code = inc.code AND fs.rpid = inc.rpid
            WHERE
                com.name = 'SamsungElec' AND fs.bsns_year BETWEEN 2016 AND 2021;
            """)
            # VIEW TABLE: Balance Sheet
            cur.execute("""
            CREATE VIEW vt_bs_005930
            AS 
            SELECT fs.bsns_year AS bsns_year, bal.quarter AS quarter, bal.account AS account, bal.value AS value
            FROM balance_sheet AS bal
            INNER JOIN company AS com
                ON com.code = bal.code
            INNER JOIN financial_statement AS fs
                ON fs.code = bal.code AND fs.rpid = bal.rpid
            WHERE
                com.name = 'SamsungElec' AND fs.bsns_year BETWEEN 2016 AND 2021;
            """)