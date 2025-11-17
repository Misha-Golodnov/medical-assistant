"""
Telegram бот для Medical Assistant
Интеграция с RAG системой для ответов на медицинские вопросы
"""

import os
import re
import logging
import asyncio
from typing import Optional
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from telegram.constants import ParseMode

# Импорт RAG системы
import sys
sys.path.append('scripts')
from rag_system import RAGSystem

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Регулярное выражение для кодов МКБ-10
ICD_PATTERN = re.compile(r'\b[IG]\d{2}(?:\.\d{1,2})?\b', re.IGNORECASE)


class MedicalAssistantBot:
    """Telegram бот медицинского ассистента"""
    
    def __init__(self, token: str):
        """
        Инициализация бота
        
        Args:
            token: Telegram Bot API токен
        """
        self.token = token
        self.rag = None
        
        # Создание приложения с увеличенным таймаутом
        from telegram.request import HTTPXRequest
        request = HTTPXRequest(
            connection_pool_size=8,
            connect_timeout=30.0,
            read_timeout=30.0,
            write_timeout=30.0,
            pool_timeout=30.0
        )
        self.application = Application.builder().token(token).request(request).build()
        
        # Регистрация обработчиков
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("find", self.find_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        logger.info("✅ Бот инициализирован")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_message = """🏥 <b>Медицинский Ассистент</b>

Привет! Я помогу найти информацию из клинических рекомендаций Минздрава РФ.

📚 <b>База знаний:</b>
• 61 клиническая рекомендация
• Коды МКБ-10: I* (сердечно-сосудистые), G* (неврология)
• 63,985 проверенных фрагментов

💬 <b>Как использовать:</b>
1. Задайте вопрос своими словами
2. Укажите код МКБ-10: /find I21
3. Получите ответ с источниками

⚠️ <b>Важно:</b>
Информация из официальных клинических рекомендаций для специалистов.
Окончательные решения принимает лечащий врач!

📖 <b>Команды:</b>
/help - справка
/find &lt;код&gt; - поиск по коду МКБ-10
"""
        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.HTML
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_message = """📖 <b>Справка</b>

<b>Примеры вопросов:</b>
• Диагностика инфаркта миокарда
• Лечение эпилептического статуса
• Показания к госпитализации при инсульте
• Методы реваскуляризации при ОКС

<b>Поиск по МКБ-10:</b>
/find I21 - инфаркт миокарда
/find G40 - эпилепсия
/find I63 - инфаркт мозга

<b>Автоопределение кодов:</b>
Если в вопросе есть код МКБ-10 (например, "что такое I21.0"), я автоматически найду релевантные рекомендации.

⚡️ <b>Время ответа:</b> 3-10 секунд
"""
        await update.message.reply_text(
            help_message,
            parse_mode=ParseMode.HTML
        )
    
    async def find_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /find <код>"""
        if not context.args:
            await update.message.reply_text(
                "❌ Укажите код МКБ-10\n\nПример: /find I21"
            )
            return
        
        icd_code = context.args[0].upper()
        
        # Валидация кода
        if not re.match(r'^[IG]\d{2}(?:\.\d{1,2})?$', icd_code):
            await update.message.reply_text(
                f"❌ Неверный формат кода МКБ-10: {icd_code}\n\n"
                "Примеры правильных кодов:\n"
                "• I21 (инфаркт миокарда)\n"
                "• I21.0 (уточненный код)\n"
                "• G40 (эпилепсия)"
            )
            return
        
        # Отправка статуса
        status_msg = await update.message.reply_text(
            f"🔍 Ищу информацию по коду {icd_code}..."
        )
        
        try:
            # Поиск по коду МКБ-10
            query = f"Информация по коду МКБ-10 {icd_code}"
            result = await self.search_and_answer(
                query=query,
                filter_icd_codes=[icd_code]
            )
            
            # Форматирование и отправка ответа
            await status_msg.delete()
            await self.send_formatted_answer(update, result)
            
        except Exception as e:
            logger.error(f"Ошибка обработки /find {icd_code}: {e}")
            await status_msg.edit_text(
                f"❌ Произошла ошибка при поиске информации по коду {icd_code}"
            )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        query = update.message.text.strip()
        
        if not query:
            return
        
        # Автоопределение кодов МКБ-10
        icd_codes = ICD_PATTERN.findall(query)
        icd_codes = [code.upper() for code in icd_codes] if icd_codes else None
        
        # Проверка: есть ли медицинский контекст
        medical_keywords = [
            'лечение', 'диагностика', 'симптом', 'терапия', 'препарат',
            'заболевание', 'болезнь', 'синдром', 'инфаркт', 'инсульт',
            'гипертония', 'диабет', 'мигрень', 'эпилепсия', 'рекоменд',
            'пациент', 'анализ', 'обследование', 'клиническ', 'mkb', 'мкб'
        ]
        
        has_medical_context = (
            icd_codes is not None or 
            any(keyword in query.lower() for keyword in medical_keywords)
        )
        
        if not has_medical_context:
            await update.message.reply_text(
                "❓ Не могу понять ваш вопрос.\n\n"
                "Я помогу вам с информацией по клиническим рекомендациям.\n\n"
                "Примеры вопросов:\n"
                "• /find I21 - информация по коду МКБ-10\n"
                "• Лечение инфаркта миокарда I21\n"
                "• Диагностика инсульта G40\n\n"
                "Или используйте /help для справки"
            )
            return
        
        # Отправка статуса
        status_msg = await update.message.reply_text("🤔 Думаю...")
        
        try:
            # Поиск и генерация ответа
            result = await self.search_and_answer(
                query=query,
                filter_icd_codes=icd_codes,
                user_id=update.effective_user.id,
                username=update.effective_user.username
            )
            
            # Форматирование и отправка ответа
            await status_msg.delete()
            await self.send_formatted_answer(update, result)
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса '{query}': {e}")
            await status_msg.edit_text(
                "❌ Произошла ошибка при обработке запроса. Попробуйте еще раз."
            )
    
    async def search_and_answer(
        self,
        query: str,
        filter_icd_codes: Optional[list] = None,
        user_id: Optional[int] = None,
        username: Optional[str] = None
    ) -> dict:
        """
        Поиск и генерация ответа через RAG
        
        Args:
            query: Запрос пользователя
            filter_icd_codes: Фильтр по кодам МКБ-10
            user_id: ID пользователя Telegram
            username: Username пользователя
            
        Returns:
            Результат RAG pipeline
        """
        # Инициализация RAG (если еще не инициализирована)
        if self.rag is None:
            self.rag = RAGSystem()
            self.rag.connect()
        
        # Выполнение RAG pipeline
        result = self.rag.ask(
            query=query,
            n_results=50,
            similarity_threshold=0.1,
            temperature=0.3,
            filter_icd_codes=filter_icd_codes,
            user_id=user_id,
            username=username
        )
        
        return result
    
    async def send_formatted_answer(self, update: Update, result: dict):
        """
        Форматирование и отправка ответа
        
        Args:
            update: Telegram Update объект
            result: Результат RAG pipeline
        """
        # Основной ответ - экранируем HTML символы
        answer = result['answer'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Форматирование источников в HTML
        sources_text = "\n\n📚 <b>Источники:</b>\n"
        for idx, source in enumerate(result['sources'], 1):
            icd_str = ", ".join(source['icd_codes']) if source['icd_codes'] else "—"
            # Экранируем HTML символы в тексте
            title_escaped = source['title'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            sources_text += f"\n{idx}. <a href='{source['url']}'>{title_escaped}</a>\n"
            sources_text += f"   МКБ-10: {icd_str}\n"
        
        # Дисклеймер
        disclaimer = "\n\n⚠️ <i>Информация из клинических рекомендаций для специалистов. Не является руководством к самолечению.</i>"
        
        # Объединение
        full_message = answer + sources_text + disclaimer
        
        # Отправка (разбиение на части если слишком длинное)
        if len(full_message) > 4096:
            # Отправляем ответ отдельно
            await update.message.reply_text(
                answer,
                parse_mode=ParseMode.HTML
            )
            # Отправляем источники отдельно
            await update.message.reply_text(
                sources_text + disclaimer,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
        else:
            await update.message.reply_text(
                full_message,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
    
    @staticmethod
    def escape_markdown(text: str) -> str:
        """
        Экранирование спецсимволов для Markdown V2
        
        Args:
            text: Исходный текст
            
        Returns:
            Экранированный текст
        """
        # Все спецсимволы Markdown V2
        escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in escape_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    async def run_async(self):
        """Асинхронный запуск бота"""
        logger.info("🚀 Запуск Telegram бота...")
        
        # Инициализация приложения
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        
        # Ожидание остановки
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            logger.info("⏸️  Получен сигнал остановки")
        finally:
            # Остановка бота
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
    
    def run(self):
        """Запуск бота"""
        asyncio.run(self.run_async())
    
    def stop(self):
        """Остановка бота и закрытие соединений"""
        if self.rag:
            self.rag.close()
        logger.info("🛑 Бот остановлен")


def main():
    """Главная функция"""
    # Получение токена
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN не найден в .env")
        return
    
    # Создание и запуск бота
    bot = MedicalAssistantBot(token)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("⏸️  Получен сигнал остановки")
    finally:
        bot.stop()


if __name__ == "__main__":
    main()
