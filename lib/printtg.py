import asyncio
import logging
from telegram import Bot
from telegram.constants import ParseMode


class TelegramLoggerHandler(logging.Handler):
    def __init__(self, bot_token, chat_id):
        super().__init__()
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=self.bot_token)

    def emit(self, record):
        log_entry = self.format(record)
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self.send_telegram_message(log_entry))

    async def send_telegram_message(self, message):
        await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode=ParseMode.HTML)


def print_tg(message):
    bot_token = "7281018218:AAHKjvghm08QvNZPyME1D6C16Sxla-HAB-c"
    chat_id = "334235401"

    logger = logging.getLogger("logger")
    logger.setLevel(logging.WARNING)

    # Prevent adding multiple handlers if print_tg is called multiple times
    if not any(isinstance(handler, TelegramLoggerHandler) for handler in logger.handlers):
        telegram_handler = TelegramLoggerHandler(bot_token=bot_token, chat_id=chat_id)
        logger.addHandler(telegram_handler)

    logger.warning(message)