import sys

import pandas as pd


class ClientManager:
    def __init__(self, dbm, qc_clients_tbl):
        self.dbm = dbm
        self.qc_clients_tbl = qc_clients_tbl

    def add_client(self, client_id, client_name, client_surname, client_midname, client_gender, client_email, manager_id):
        df = pd.DataFrame({'client_id': [client_id],
                           'client_name': [client_name],
                           'client_surname': [client_surname],
                           'client_midname': [client_midname],
                           'client_gender': [client_gender],
                           'client_email': [client_email],
                           'manager_id': [manager_id]})
        self.dbm.insert_df_fast(df, self.qc_clients_tbl)
        self.dbm.commit()

    def remove_client(self, client_id):
        self.dbm.delete_table_row(self.qc_clients_tbl, {'client_id': client_id})
        self.dbm.commit()

    def get_df(self):
        df = self.dbm.select_df(f"select * from {self.qc_clients_tbl}")
        return df

    def __str__(self):
        df = self.dbm.select_df(f"select * from {self.qc_clients_tbl}")
        return str(df)


if __name__ == '__main__':
    sys.path[0] = '../../..'
    import config
    from QuantumCapital import constants
    from QuantumCapital.DBManager import DBManager

    DB_USERNAME = config.DB_USERNAME
    DB_PASS = config.DB_PASS

    QC_MANAGER_TABLE = constants.QC_MANAGERS_TABLE
    QC_CLIENTS_TABLE = constants.QC_CLIENTS_TABLE

    dbm = DBManager(DB_USERNAME, DB_PASS, 'ODS')

    manager = ClientManager(dbm, QC_CLIENTS_TABLE)
    # manager.add_client(122, 'Талгат', 'Муслимов', 'Александрович', 'm', 'talmus@gmail.com', 1)
    # manager.add_client(129, 'Айгерим', 'Шалкарова', 'Сериккызы', 'f', 'arna2@gmail.com', 1)
    # manager.add_client(130, 'Арнур', 'Садвакасов', 'Айбарулы', 'm', 'arna@gmail.com', 2)
    print(manager)
