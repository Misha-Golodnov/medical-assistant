# Medical Assistant Bot

Медицинский ассистент на базе RAG (Retrieval-Augmented Generation) для работы с клиническими рекомендациями Минздрава РФ.

## 🎯 Возможности

- 📚 База знаний: 60 клинических рекомендаций (коды МКБ-10: I*, G*)
- 🔍 Векторный поиск по 62,879 семантическим чанкам
- 🤖 Генерация детальных клинических ответов через GPT-4o (расширенный контекст: 50 chunks, 32K tokens)
- 💬 Telegram бот с медицинской фильтрацией и HTML-форматированием
- 📊 Логирование запросов в PostgreSQL с аналитикой стоимости
- 🔎 Умный поиск по кодам МКБ-10 (I21 → I21.0, I21.1, ...)

## 🗄️ Технологический стек

- **База данных**: PostgreSQL 16 + pgvector 0.8.1
- **Embeddings**: OpenAI text-embedding-3-large (3072 размерности)
- **LLM**: GPT-4o
- **Chunking**: Chonkie SemanticChunker
- **Фреймворк**: Python 3.13

## 📊 Статистика

- **Документов**: 60 клинических рекомендаций (1 файл поврежден: 739_2)
- **Chunks**: 62,879 семантических блоков
- **Embeddings**: 62,879 векторов (3072-мерных)
- **Размер БД**: 875 MB
- **Стоимость генерации**: $1.40 (embeddings) + ~$0.022/запрос (GPT-4o с расширенным контекстом)
- **Производительность**: 0.4 сек на запрос (B-tree индексы), кэширование частых запросов

## 🚀 Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd medical-assistant
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка переменных окружения

Создайте файл `.env`:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL
DATABASE_URL=postgresql://postgres:postgres123@localhost:5433/medical_assistant

# Telegram Bot (опционально)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

### 5. Запуск PostgreSQL + pgvector

```bash
docker-compose up -d
```

База данных автоматически инициализируется через `init.sql` при первом запуске контейнера.



## 📁 Структура проекта

```
medical-assistant-v2/
├── scripts/               # Core модули
│   ├── rag_system.py     # RAG система с расширенным контекстом
│   └── view_logs.py      # Просмотр логов запросов из PostgreSQL
├── bot.py                 # Telegram бот
├── docker-compose.yml     # Docker конфигурация (PostgreSQL + pgvector)
├── init.sql              # SQL инициализация (таблицы + индексы)
├── requirements.txt      # Python зависимости
├── .env                  # Переменные окружения (API ключи)
└── README.md            # Документация
```

**База данных PostgreSQL (875 MB):**
- `chunks` - 62,879 семантических блоков с embeddings (3072-dim)
- `cr_metadata` - 60 клинических рекомендаций с метаданными
- `query_logs` - логи всех запросов с метриками и стоимостью

## 🔧 Использование

### RAG система (standalone)

```python
from scripts.rag_system import RAGSystem

# Инициализация
with RAGSystem() as rag:
    # Поиск и генерация ответа с расширенным контекстом
    result = rag.ask(
        query="Какие методы диагностики миокардита?",
        n_results=50,              # Расширенный контекст для врачей
        similarity_threshold=0.1,   # Низкий порог для большего охвата
        user_id=123456,            # ID пользователя для логирования
        username="doctor_ivanov"   # Username для аналитики
    )
    
    print(result['answer'])
    print(f"\nИсточники: {len(result['sources'])}")
    print(f"Токенов использовано: {result['tokens']['total']}")
    for source in result['sources']:
        print(f"- {source['title']} (similarity: {source['similarity']:.3f})")
```

### Telegram бот

```bash
python bot.py
```

**Команды бота:**
- `/start` - Приветствие и инструкции
- `/help` - Справка по использованию
- `/find <код>` - Поиск по коду МКБ-10 (например: `/find I21`)
- Текстовые запросы - Автоматическое определение МКБ кодов и медицинского контекста

**Особенности:**
- Медицинская фильтрация (отсекает нерелевантные запросы)
- HTML-форматирование ответов с экранированием
- Поддержка базовых кодов МКБ (I21 → I21.0, I21.1, ...)
- Детальные клинические ответы для врачей

### Просмотр логов запросов

```bash
# Последние 20 запросов за 24 часа
python scripts/view_logs.py

# Последние 50 запросов
python scripts/view_logs.py 50

# Статистика по пользователям
python scripts/view_logs.py stats

# Все запросы за неделю
python scripts/view_logs.py all
```

## 🛠️ Архитектура системы

**RAG Pipeline:**
1. **Векторный поиск** - PostgreSQL + pgvector (cosine similarity, B-tree индексы)
2. **Генерация embedding запроса** - OpenAI text-embedding-3-large (3072-dim)
3. **Фильтрация по МКБ-10** - Умный поиск с prefix matching (I21 → I21.*)
4. **Ранжирование** - Top-50 chunks по similarity (порог 0.1)
5. **Генерация ответа** - GPT-4o с расширенным контекстом (32K tokens)
6. **Логирование** - PostgreSQL query_logs (метрики + стоимость)

## 📈 Производительность

### Генерация данных
- **Chunking**: ~46 минут, $0.85
- **Embeddings**: 22 минуты, $1.40
- **Database load**: ~4 минуты (62,879 chunks)

### Runtime производительность
- **Query latency**: 3-10 секунд (50 chunks + генерация ответа)
  - Поиск по векторам (B-tree): 0.4 сек
  - Генерация embedding запроса: 1-2 сек
  - GPT-4o генерация: 5-20 сек (зависит от длины ответа)
- **Кэширование**: Частые запросы кэшируются (до 100 записей)
- **Стоимость запроса**: ~$0.022 (embedding: $0.000001 + GPT input: $0.015 + output: $0.007)

### База данных
- **Индексы**: B-tree на cr_id, icd_codes, created_at (pgvector ограничение: макс 2000 dim для HNSW)
- **Размер**: 875 MB (chunks + embeddings + метаданные + логи)
- **Backup**: JSON файлы chunks/embeddings сохранены для восстановления

## 🔐 Безопасность

- API ключи в `.env` (не коммитятся)
- PostgreSQL пароли в переменных окружения
- Медицинский дисклеймер в ответах бота
- Медицинская фильтрация запросов (20+ ключевых слов)
- HTML-экранирование для защиты от XSS в Telegram

## 📊 Логирование и аналитика

Все запросы логируются в PostgreSQL таблицу `query_logs` с детализацией:
- Параметры запроса (query, user_id, username, n_results, similarity_threshold)
- Метрики поиска (top_similarity, results_count, context_chars, context_crs)
- Использование токенов (prompt_tokens, completion_tokens, total_tokens)
- Стоимость (embedding_cost, input_cost, output_cost, total_cost)
- Метаданные (cached, response_length, execution_time_ms, created_at)

Используйте `scripts/view_logs.py` для анализа и отчетности.

## 🚀 Деплой на сервер

Проект готов для деплоя через Docker на VPS. Подробная инструкция в разделе документации.

**Системные требования VPS:**
- CPU: 2+ ядра
- RAM: 4GB (PostgreSQL + векторы)
- SSD: 20GB (база ~1GB + логи)
- OS: Ubuntu 22.04 LTS

**Рекомендуемые провайдеры:**
- Hetzner CX21: €5.83/мес
- DigitalOcean: $12/мес
- Contabo VPS-S: €5.50/мес

## 🔄 Обновления

### v2.0 (текущая версия)
- ✅ Telegram бот с командой `/find`
- ✅ Расширенный контекст для врачей (50 chunks, 32K tokens)
- ✅ Логирование в PostgreSQL
- ✅ HTML-экранирование в ответах
- ✅ Умный поиск по базовым кодам МКБ (I21 → I21.*)
- ✅ Медицинская фильтрация запросов
- ✅ Кэширование частых запросов
- ✅ Увеличенные таймауты для Telegram API

