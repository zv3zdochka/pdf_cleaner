from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from aiogram import BaseMiddleware
from aiogram.types import Message, CallbackQuery, TelegramObject, KeyboardButton, ReplyKeyboardMarkup, \
    ReplyKeyboardRemove

from pdf_cleaner_bot.storage.user_db import UserDatabase


async def handle_contact(message: Message, user_db: UserDatabase) -> None:
    """Handle contact message - save phone number."""
    log = logging.getLogger("pdf_cleaner.contact_handler")
    log.info("handle_contact called for user %s", message.from_user.id if message.from_user else "unknown")

    user = message.from_user
    contact = message.contact

    if not user:
        log.warning("No user in message")
        return

    if not contact:
        log.warning("No contact in message")
        return

    log.info("Contact received: user_id=%s, contact.user_id=%s, phone=%s",
             user.id, contact.user_id, contact.phone_number)

    if contact.user_id != user.id:
        await message.answer("Пожалуйста, отправьте свой собственный контакт.")
        return

    user_db.update_phone_number(user.id, contact.phone_number)
    log.info("Phone number saved for user %s", user.id)

    await message.answer(
        "✅ Спасибо! Ваш номер телефона сохранён.\n\n"
        "Теперь вы можете отправить PDF-файл для обработки.",
        reply_markup=ReplyKeyboardRemove()
    )


class UserTrackingMiddleware(BaseMiddleware):

    def __init__(self, user_db: UserDatabase, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.user_db = user_db
        self.log = logger or logging.getLogger(__name__)

    async def _send_request_contact(self, message: Message) -> None:
        button = KeyboardButton(
            text="📱 Поделиться номером телефона",
            request_contact=True
        )
        keyboard = ReplyKeyboardMarkup(
            keyboard=[[button]],
            resize_keyboard=True,
            one_time_keyboard=True
        )
        await message.answer(
            "Для продолжения работы, пожалуйста, поделитесь своим номером телефона:",
            reply_markup=keyboard,
        )

    async def __call__(
            self,
            handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
            event: TelegramObject,
            data: Dict[str, Any],
    ) -> Any:
        user = None
        chat = None
        is_pdf = False

        if isinstance(event, Message):
            user = event.from_user
            chat = event.chat

            self.log.info("Middleware: Message from user %s, has_contact=%s, content_type=%s",
                          user.id if user else None,
                          event.contact is not None,
                          event.content_type)

            # Let contact messages through to their handler
            if event.contact is not None:
                self.log.info("Middleware: Passing contact message to handler")
                return await handler(event, data)

            # Gatekeeping: require phone number in private chats
            if user and chat and chat.type == "private":
                has_phone = self.user_db.has_phone_number(user.id)
                self.log.info("Middleware: User %s has_phone=%s", user.id, has_phone)

                if not has_phone:
                    try:
                        self.user_db.record_user_activity(
                            user_id=user.id,
                            username=user.username,
                            first_name=user.first_name,
                            last_name=user.last_name,
                            language_code=user.language_code,
                            is_bot=user.is_bot,
                            chat_id=chat.id,
                            chat_type=chat.type,
                            is_pdf=False,
                            phone_number=None
                        )
                    except Exception as e:
                        self.log.error("Failed to init user record: %s", e)

                    await self._send_request_contact(event)
                    return

            if event.document and event.document.mime_type == "application/pdf":
                is_pdf = True

        elif isinstance(event, CallbackQuery):
            user = event.from_user
            if event.message:
                chat = event.message.chat

        # Track user activity
        if user is not None:
            try:
                self.user_db.record_user_activity(
                    user_id=user.id,
                    username=user.username,
                    first_name=user.first_name,
                    last_name=user.last_name,
                    language_code=user.language_code,
                    is_bot=user.is_bot,
                    chat_id=chat.id if chat else None,
                    chat_type=chat.type if chat else None,
                    is_pdf=is_pdf,
                    phone_number=None
                )
            except Exception as e:
                self.log.error("Failed to track user %s: %s", user.id, e)

        return await handler(event, data)