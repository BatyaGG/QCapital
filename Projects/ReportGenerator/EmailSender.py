from MergePDFs import merge_pdfs

from ManagerManager import ManagerManager
from ClientManager import ClientManager
from TitlePageGen import TitlePageGen

import sys, os
import email, smtplib, ssl

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sys.path[0] = '../../..'
import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager


DB_USERNAME = config.DB_USERNAME
DB_PASS = config.DB_PASS

QC_MANAGER_TABLE = constants.QC_MANAGERS_TABLE
QC_CLIENTS_TABLE = constants.QC_CLIENTS_TABLE

dbm = DBManager(DB_USERNAME, DB_PASS, 'ODS')


sender_email = "batyrkhan.saduanov@gmail.com"
password = input("Type your password and press enter:")
title_page_text_path = 'Reports/title_text.txt'
report_path = 'Reports/Годовой отчет 2020.pdf'


with open(title_page_text_path, 'r') as file:
    title_page_text = file.read().replace('\n', '')

mng_mng = ManagerManager(dbm, QC_MANAGER_TABLE)
managers = mng_mng.get_df()

clnt_mng = ClientManager(dbm, QC_CLIENTS_TABLE)
clients = clnt_mng.get_df()

for i in range(managers.shape[0]):
    manager = managers.iloc[i]
    curr_clients = clients[clients.manager_id == manager.manager_id]

    subject = "Отчеты для клиентов"
    body = "Во вложении отчеты для следующих клиентов:\n\n" + str(curr_clients[['client_id',
                                                                                'client_name',
                                                                                'client_surname',
                                                                                'client_midname',
                                                                                'client_email']])
    receiver_email = manager.manager_email

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails
    message.attach(MIMEText(body, "plain"))

    final_reports = []

    for i in range(curr_clients.shape[0]):
        curr_client = curr_clients.iloc[i]
        temp_file_name = 'temp_title_page.pdf'
        title_page_gen = TitlePageGen(temp_file_name, config.PROJ_DIR)
        title_page_gen.main_drawer(curr_client.client_name,
                                   curr_client.client_midname,
                                   curr_client.client_gender,
                                   title_page_text)
        title_page_gen.save()
        final_report_path = f'{curr_client.client_name[0]}_{curr_client.client_surname}_report.pdf'
        final_reports.append(final_report_path)
        merge_pdfs(temp_file_name, report_path, final_report_path)

        with open(final_report_path, "rb") as attachment:
            # Add file as application/octet-stream
            # Email client can usually download this automatically as attachment
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)

        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename=final_report_path)

        # Add attachment to message and convert message to string
        message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

#    for report in final_reports:
#        os.remove(report)
