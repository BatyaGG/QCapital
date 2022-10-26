import sys
import warnings
import re

from io import StringIO

import psycopg2 as pg
import pandas as pd
import pandas.io.sql as psql
import psycopg2.extras as extras


sys.path[0] = '..'
import config


class DBManager(object):
    def __init__(self, user, pswd, db, host=config.SERVER_IP, port=config.DB_PORT):
        self.connection = pg.connect(user=user,
                                     password=pswd,
                                     database=db,
                                     host=host,
                                     port=port)

    def __del__(self):
        self.connection.close()

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def _get_column_names(self, table):
        cursor = self.connection.cursor()
        try:
            cursor.execute(f'select * from {table.lower()} limit 0')
            col_names = [desc[0] for desc in cursor.description]
            cursor.close()
        except pg.DatabaseError as error:
            raise pg.DatabaseError(error)  # you didn't
        return col_names

    def insert_df_fast(self, df: pd.DataFrame, table: str):
        warnings.warn('If you have datetime field, you should use timezone aware object only!')
        cols = self._get_column_names(table)
        if len(cols) != df.shape[1]:
            raise Exception(f'Table and DF column number must match\n'
                            f'Table col num: {len(cols)}\nDF col num: {df.shape[1]}')
        elif cols != list(df.columns):
            raise Exception(f'Table and DF column names must match and be in proper order\n'
                            f'Proper order: {cols}')
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        cursor = self.connection.cursor()
        try:
            cursor.copy_from(buffer, table.lower(), sep=",", null='null')
            # self.connection.commit()
            cursor.close()
        except pg.DatabaseError as error:
            # self.connection.rollback()
            cursor.close()
            raise pg.DatabaseError(error)

    def insert_df_simple(self, df: pd.DataFrame, table: str):
        # warnings.warn('If you have datetime field, you should use timezone aware object only!')

        tuples = [tuple(x) for x in df.to_numpy()]
        # Comma-separated dataframe columns
        cols = ','.join(list(df.columns))
        # SQL query to execute
        query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
        cursor = self.connection.cursor()
        try:
            extras.execute_values(cursor, query, tuples)
            # self.connection.commit()
        except (Exception, pg.DatabaseError) as error:
            print("Error: %s" % error)
            # self.connection.rollback()
            cursor.close()
            raise pg.DatabaseError(error)

        cursor.close()

    def select_df(self, select_str: str) -> pd.DataFrame:
        return psql.read_sql(select_str, self.connection)

    def select_cursor(self, select_str: str):
        cursor = self.connection.cursor()
        cursor.execute(select_str)
        return cursor

    def truncate_table(self, name: str):
        cursor = self.connection.cursor()
        try:
            cursor.execute("TRUNCATE TABLE {} RESTART IDENTITY".format(name.lower()))
            # self.connection.commit()
            cursor.close()
        except pg.DatabaseError as error:
            # self.connection.rollback()
            cursor.close()
            raise pg.DatabaseError(error)

    def create_table(self, name: str, columns: dict):
        """
        :param name: table name
        :param columns: dict with keys as column names and values as their types
        """
        # warnings.warn('If you have datetime field, you should use \"timestamp with time zone\" type only!')
        cursor = self.connection.cursor()
        try:
            cursor.execute("CREATE TABLE {} ({})".format(name.lower(), ', '.join([f'{k} {columns[k]}' for k in columns])))
            # self.connection.commit()
            cursor.close()
        except pg.DatabaseError as error:
            # self.connection.rollback()
            cursor.close()
            raise pg.DatabaseError(error)

    def check_if_table_exists(self, name):
        # is_exists = True
        # cursor = self.connection.cursor()
        # try:
        #     cursor.execute(f'select * from {name} limit 0')
        # except pg.errors.lookup('42P01'):
        #     is_exists = False
        # finally:
        #     cursor.close()
        #     # TODO: avoid this rollback
        #     self.connection.rollback()
        try:
            df = self.select_df(f'select * from {name} limit 0')
            return True
        except:
            return False

    def drop_table(self, name: str or list):
        """

        :param name: str or list of table names
        :return:
        """
        cursor = self.connection.cursor()
        try:
            if type(name) is str:
                cursor.execute("DROP TABLE {}".format(name.lower()))
            elif type(name) is list:
                cursor.execute("DROP TABLE {}".format(', '.join([n.lower() for n in name])))
            # self.connection.commit()
            cursor.close()
        except pg.DatabaseError as error:
            # self.connection.rollback()
            cursor.close()
            raise pg.DatabaseError(error)

    def drop_by_name_start(self, match_str: str):
        cursor = self.select_cursor('select table_name from information_schema.tables where table_schema = \'public\'')
        for table in cursor:
            table = table[0]
            match = re.match(r'{}'.format(match_str), table)
            if match is not None:
                print(table)
                self.drop_table(table)
        cursor.close()

    def update_table_row(self, table: str, set_cols: dict, where_cols: dict, add_params: tuple=None):
        """
        Updates table rows by condition. Be careful - if there is no matching rows
        to update -> no exception will be raised. You should check it manually in your code.

        :param table: table name
        :param set_cols: columns to update with update values as keys
        :param where_cols: columns for where clause with condition values as keys
        :return: None
        """
        # SQL = f"update {table} " \
        #       f"set {', '.join([     f' {col} = {set_cols[col]} '       for col in set_cols       ])} " \
        #       f"where {' and '.join([f'{col}={where_cols[col]}' for col in where_cols])}"

        SQL = f"update {table} " \
              f"set {', '.join([     f' {col}={set_cols[col]} '       for col in set_cols       ])} " \
              f"where {' and '.join([f'{col} {where_cols[col]}' for col in where_cols])}"
        cursor = self.connection.cursor()
        if add_params is None:
            cursor.execute(SQL)
        else:
            print(SQL)
            cursor.execute(SQL, add_params)
        # self.connection.commit()
        cursor.close()
        f"asdasdas \' asdasdasd \'{5}"

    def delete_table_row(self, table: str, where_cols: dict):
        """
        Updates table rows by condition. Be careful - if there is no matching rows
        to update -> no exception will be raised. You should check it manually in your code.

        :param table: table name
        :param set_cols: columns to update with update values as keys
        :param where_cols: columns for where clause with condition values as keys
        :return: None
        """
        SQL = f"delete from {table} " \
              f"where {' and '.join([f'{col}={where_cols[col]}' for col in where_cols])}"
        cursor = self.connection.cursor()
        cursor.execute(SQL)
        # self.connection.commit()
        cursor.close()
        f"asdasdas \' asdasdasd \'{5}"


if __name__ == '__main__':
    from datetime import datetime
    import pytz

    # db_manager_bdy = DBManager('Batyrkhan', 'asdasd', 'BARS_DAY')
    db_manager_ods = DBManager('Batyrkhan', 'asdasd', 'ODS')

    # for i in range(2005, 2022):
    #     ods = db_manager_ods.select_df(f'select * from bars_1_day_1russel_presence where yr = {i}').iloc[0]
    #     bdy = db_manager_bdy.select_df(f'select * from bars_1_day_1russel_presence where yr = {i}').iloc[0]
    #
    #     print(len(ods.present), len(bdy.present))
    #     print(len(ods.absent), len(bdy.absent))
    #     print('-----------------------------------------------------------\n')
    # df = db_manager.select_df('select * from qeqeqweqweqwrqwrsad')
    # print(df)
    # df.to_csv('wsb_comments.csv', index=False)
    # for i in range(df.shape[0]):
    #     row = df.iloc[i, :]
    #     yr = row.yr
    #     tickers = row.tickers
    #     print(yr, len(tickers))
    # df = db_manager.select_df('select * from wsb_discussion_thread_comments')
    # print(df)
    # df = db_manager.select_df('select * from reporting_in_2weeks_tickers')
    # db_manager.truncate_table('ods_test')
    # cols = {'dt': 'timestamp with time zone',
    #         'open': 'real',
    #         'high': 'real',
    #         'low': 'real',
    #         'close': 'real',
    #         'volume': 'real'}
    # db_manager.create_table('bars_1_min_aa', cols)
    db_manager_ods.drop_by_name_start('bars_1_day_[a-zA-Z1-9]')
    db_manager_ods.commit()
    # db_manager.commit()

    # cols = {'table_name': '\'ASDASD\'', f'dates': f"array_append(dates, timestamp with time zone \'{datetime(2020, 2, 5, 23, 0, 0, 0, pytz.UTC)}\')"}
    # print(cols)
    # where = {'bars_count': 555, 'memory': 50}
    # db_manager.update_table_row('bars_1_min_1dictionary', cols, where)

    # df = {'dt': [datetime(2000, 1, 1, 0, 0, 0)], 'open': [10], 'high': [10], 'low': [10], 'close': [10], 'volume': [10]}
    #
    # db_manager.insert_df_fast(pd.DataFrame(df), 'aaa')

    # df = db_manager.select_df('select * from bars_1_min_1russel_presence')
    # print(df.shape)

    # df = db_manager.select_df('select * from bars_1_day_1russel_presence')
    # print(df)
    # i = 7
    # print(df.iloc[i].yr)
    # present = set(df.iloc[i].present)
    # absent = set(df.iloc[i].absent)
    # intersect = present & absent
    # print(len(present), len(absent), len(intersect))

    # df = db_manager.select_df('select * from bars_1_min_noc order by dt desc')
    # last_date = df.iloc[0]['dt']
    # print(df.shape)
    # db_manager.delete_table_row('bars_1_min_noc', {'dt': "\'" + str(last_date) + "\'"})
    # db_manager.commit()

    # db_manager.truncate_table('reporting_small_tickers')
    # db_manager.commit()
