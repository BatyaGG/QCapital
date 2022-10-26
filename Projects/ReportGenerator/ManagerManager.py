import sys

import pandas as pd

# sys.path[0] = '../..'
# import config
# from QuantumCapital import constants
# from QuantumCapital.DBManager import DBManager


class ManagerManager:
    def __init__(self, dbm, qc_manager_tbl):
        self.dbm = dbm
        self.qc_manager_tbl = qc_manager_tbl

    def add_manager(self, manager_id, name, surname, email):
        df = pd.DataFrame({'manager_id': [manager_id],
                           'manager_name': [name],
                           'manager_surname': [surname],
                           'manager_email': [email]})
        self.dbm.insert_df_fast(df, self.qc_manager_tbl)
        self.dbm.commit()

    def remove_manager(self, manager_id):
        self.dbm.delete_table_row(self.qc_manager_tbl, {'manager_id': manager_id})
        self.dbm.commit()

    def get_df(self):
        df = self.dbm.select_df(f"select * from {self.qc_manager_tbl}")
        return df

    def __str__(self):
        df = self.dbm.select_df(f"select * from {self.qc_manager_tbl}")
        return str(df)


if __name__ == '__main__':
    manager = ManagerManager()
    # manager.add_manager(1, 'Batyrkhan', 'Saduanov', 'batyrkhan.saduanov@nu.edu.kz')
    # manager.add_manager(2, 'Alibi', 'Zhangeldin', 'alibi.jangeldin@nu.edu.kz')
    # manager.remove_manager(1)
    print(manager)
