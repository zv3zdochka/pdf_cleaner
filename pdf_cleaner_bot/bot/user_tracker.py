"""User tracking middleware for aiogram.

This module provides a middleware that captures Telegram user data
from every incoming update and records it to the user database.

The middleware is non-intrusive and does not modify the behavior
of existing handlers.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from aiogram import BaseMiddleware
from aiogram.types import Message, CallbackQuery, TelegramObject

from pdf_cleaner_bot.storage.user_db import UserDatabase


class UserTrackingMiddleware(BaseMiddleware):
    """Middleware that tracks user activity on every update."""

    def __init__(self, user_db: UserDatabase, logger: Optional[logging.Logger] = None):
        """Initialize the middleware.

        Parameters
        ----------
        user_db:
            UserDatabase instance for persisting user data.
        logger:
            Optional logger instance.
        """
        super().__init__()
        self.user_db = user_db
        self.log = logger or logging.getLogger(__name__)
        self.log.info("UserTrackingMiddleware initialized")

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        """Process update and track user before passing to handler."""
        # Extract user and chat info based on event type
        user = None
        chat = None
        is_pdf = False
        phone_number = None

        if isinstance(event, Message):
            user = event.from_user
            chat = event.chat

            # Check if this is a PDF document
            if event.document and event.document.mime_type == "application/pdf":
                is_pdf = True
                self.log.info("Detected PDF upload from user %s", user.id if user else "unknown")

            # Check if user shared contact with phone number
            if event.contact and user and event.contact.user_id == user.id:
                phone_number = event.contact.phone_number
                self.log.info("User %s shared phone number", user.id)

        elif isinstance(event, CallbackQuery):
            user = event.from_user
            if event.message:
                chat = event.message.chat

        # Record user activity if we have user info
        if user is not None:
            self.log.info(
                "Tracking user: id=%s, username=%s, first_name=%s, chat_id=%s",
                user.id, user.username, user.first_name,
                chat.id if chat else None
            )
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
                    phone_number=phone_number,
                )
                self.log.info("Successfully tracked user %s", user.id)
            except Exception as e:
                self.log.error("Failed to track user %s: %s", user.id, e, exc_info=True)
        else:
            self.log.debug("No user info in event: %s", type(event).__name__)

        # Continue to the actual handler
        return await handler(event, data)