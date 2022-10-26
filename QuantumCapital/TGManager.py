import sys
from typing import Union, List

import telebot

import telegram
from telegram import InlineKeyboardButton
from telegram import InlineKeyboardMarkup
from telegram.ext import Updater, CallbackQueryHandler

import time


class TGManager(object):
    def __init__(self, bot_token, tws_user_id, ml_group_id, logs_group_id):
        self.bot_token = bot_token
        # self.bot = telegram.Bot(token=bot_token)
        # self.bot = telebot.TeleBot(bot_token)
        self.tws_user_id = tws_user_id
        self.ml_group_id = ml_group_id
        self.logs_group_id = logs_group_id

    def send(self, msg, which_chat):
        assert which_chat in ('tws_user', 'ml_group', 'logs_group')
        msg = '```python\n' + msg + '```'
        if which_chat == 'tws_user':
            chat_id = self.tws_user_id
        elif which_chat == 'ml_group':
            chat_id = self.ml_group_id
        else:
            chat_id = self.logs_group_id
        bot = telegram.Bot(token=self.bot_token)
        bot.sendMessage(chat_id=chat_id, text=msg, parse_mode='MarkdownV2')

    def send_file(self, url, which_chat):
        assert which_chat in ('tws_user', 'ml_group', 'logs_group')
        if which_chat == 'tws_user':
            chat_id = self.tws_user_id
        elif which_chat == 'ml_group':
            chat_id = self.ml_group_id
        else:
            chat_id = self.logs_group_id
        bot = telegram.Bot(token=self.bot_token)
        bot.sendDocument(chat_id=chat_id, document=url)

    # @staticmethod
    # def build_menu(
    #         buttons: List[InlineKeyboardButton],
    #         n_cols: int,
    #         header_buttons: Union[InlineKeyboardButton, List[InlineKeyboardButton]] = None,
    #         footer_buttons: Union[InlineKeyboardButton, List[InlineKeyboardButton]] = None
    # ) -> List[List[InlineKeyboardButton]]:
    #     menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    #     if header_buttons:
    #         menu.insert(0, header_buttons if isinstance(header_buttons, list) else [header_buttons])
    #     if footer_buttons:
    #         menu.append(footer_buttons if isinstance(footer_buttons, list) else [footer_buttons])
    #     return menu
    #
    # def tws_alert_block(self):
    #     def button(bot, update):
    #         print(update)
    #         # self.updater.dispatcher.remove_handler(handler)
    #         self.updater.is_idle = False
    #         sys.exit()
    #     msg = 'Bro, pls re-login to IB gateway as soon as possible and then click DONE'
    #     msg = '```python\n' + msg + '```'
    #     buttons = [InlineKeyboardButton("DONE", callback_data='Done')]
    #     reply_markup = InlineKeyboardMarkup(self.build_menu(buttons, n_cols=1))
    #     self.bot.sendMessage(chat_id=self.tws_user_id, text=msg, parse_mode='MarkdownV2', reply_markup=reply_markup)
    #     handler = CallbackQueryHandler(button)
    #     self.updater = Updater(self.bot_token)
    #     self.updater.dispatcher.add_handler(handler)
    #     self.updater.start_polling()
    #     self.updater.idle()

    def tws_alert_block(self):

        bot = telebot.TeleBot(self.bot_token)

        @bot.callback_query_handler(func=lambda call: True)
        def callback_inline(call):
            chat_id = call.from_user.id
            msg_id = call.message.message_id
            rpl_mrkp = call.message.reply_markup

            bot.answer_callback_query(call.id)
            if call.data == 'DONE':
                bot.send_message(chat_id, 'Good')
                bot.stop_polling()
                bot.stop_poll(chat_id, msg_id, rpl_mrkp)
                bot.stop_bot()
        msg = 'Bro, pls re-login to IB gateway as soon as possible and then click DONE'
        markup_inline = telebot.types.InlineKeyboardMarkup()
        button = telebot.types.InlineKeyboardButton(text='DONE', callback_data='DONE')
        markup_inline.add(button)

        bot.send_message(self.tws_user_id, msg, reply_markup=markup_inline)
        bot.infinity_polling(False)


if __name__ == '__main__':
    import constants
    tg_manager = TGManager(constants.QC_TOKEN,
                           constants.TWS_USER_ID,
                           constants.QC_ML_GROUP_ID,
                           constants.QC_BOT_LOGS_GROUP_ID)
    print('before alert')
    tg_manager.tws_alert_block()
    print('after alert')
    print('before alert')
    tg_manager.tws_alert_block()
    print('after alert')
