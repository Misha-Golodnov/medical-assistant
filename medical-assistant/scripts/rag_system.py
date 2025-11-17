"""
RAG система для медицинского ассистента
Векторный поиск по PostgreSQL + pgvector, генерация ответов через GPT-4o
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import lru_cache
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()


@dataclass
class SearchResult:
    """Результат векторного поиска"""
    cr_id: str
    chunk_index: int
    text: str
    similarity: float
    metadata: Dict[str, Any]
    cr_title: Optional[str] = None
    icd_codes: Optional[List[str]] = None


class RAGSystem:
    """
    RAG система для медицинского ассистента
    
    Функциональность:
    - Векторный поиск по pgvector (cosine similarity)
    - Форматирование контекста из найденных chunks
    - Генерация ответов через GPT-4o с медицинским контекстом
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        chat_model: str = "gpt-4o",
        max_context_tokens: int = 32000
    ):
        """
        Инициализация RAG системы
        
        Args:
            db_url: URL подключения к PostgreSQL
            openai_api_key: API ключ OpenAI
            embedding_model: Модель для embeddings
            chat_model: Модель для генерации ответов
            max_context_tokens: Максимальное количество токенов в контексте
        """
        self.db_url = db_url or os.getenv('DATABASE_URL')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.max_context_tokens = max_context_tokens
        
        # Инициализация OpenAI клиента
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Подключение к БД
        self.conn = None
        self.cursor = None
        
        # Кэш для частых запросов
        self._answer_cache = {}
        self._cache_max_size = 100
        
        logger.info("✅ RAGSystem инициализирован")
        logger.info(f"   Embedding model: {self.embedding_model}")
        logger.info(f"   Chat model: {self.chat_model}")
        logger.info(f"   Max context tokens: {self.max_context_tokens}")
    
    def connect(self):
        """Подключение к PostgreSQL"""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(self.db_url)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("✅ Подключение к PostgreSQL установлено")
    
    def close(self):
        """Закрытие подключения к БД"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("✅ Подключение к PostgreSQL закрыто")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Генерация embedding для запроса
        
        Args:
            query: Текст запроса
            
        Returns:
            Вектор embedding (3072 размерность для text-embedding-3-large)
        """
        try:
            response = self.openai_client.embeddings.create(
                input=query,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            logger.info(f"✅ Embedding сгенерирован (размерность: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации embedding: {e}")
            raise
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        similarity_threshold: float = 0.0,
        filter_cr_ids: Optional[List[str]] = None,
        filter_icd_codes: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Векторный поиск по базе знаний
        
        Args:
            query: Поисковый запрос
            n_results: Количество результатов
            similarity_threshold: Минимальный порог similarity (0.0 - 1.0)
            filter_cr_ids: Фильтр по конкретным CR ID
            filter_icd_codes: Фильтр по кодам МКБ-10
            
        Returns:
            Список SearchResult с релевантными chunks
        """
        logger.info(f"\n🔍 Поиск: '{query[:100]}...'")
        logger.info(f"   Результатов: {n_results}, Порог similarity: {similarity_threshold}")
        
        # Генерация embedding для запроса
        query_embedding = self.generate_query_embedding(query)
        query_vector = str(query_embedding)
        
        # Построение SQL запроса
        sql_parts = [
            """
            SELECT 
                c.cr_id,
                c.chunk_index,
                c.text,
                c.metadata,
                1 - (c.embedding <=> %s::vector) as similarity,
                m.title as cr_title,
                m.icd_codes
            FROM chunks c
            LEFT JOIN cr_metadata m ON c.cr_id = m.cr_id
            WHERE 1=1
            """
        ]
        
        params = [query_vector]
        
        # Фильтры
        if filter_cr_ids:
            sql_parts.append("AND c.cr_id = ANY(%s)")
            params.append(filter_cr_ids)
        
        if filter_icd_codes:
            # Поддержка поиска по базовому коду (I21 -> I21.*)
            # Создаем условие для каждого кода: точное совпадение ИЛИ префикс с точкой
            icd_conditions = []
            for icd_code in filter_icd_codes:
                # Если код без точки (например I21), ищем I21.*
                # Если код с точкой (например I21.0), ищем точное совпадение
                if '.' in icd_code:
                    # Точное совпадение
                    icd_conditions.append(f"%s = ANY(m.icd_codes)")
                    params.append(icd_code)
                else:
                    # Поиск по префиксу: I21 должен найти I21.0, I21.1, и т.д.
                    # Используем EXISTS с unnest для проверки префикса
                    icd_conditions.append(
                        f"EXISTS (SELECT 1 FROM unnest(m.icd_codes) AS code WHERE code LIKE %s)"
                    )
                    params.append(f"{icd_code}.%")
            
            if icd_conditions:
                sql_parts.append(f"AND ({' OR '.join(icd_conditions)})")
        
        # Сортировка и лимит
        sql_parts.append(
            f"""
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """
        )
        params.extend([query_vector, n_results])
        
        sql_query = " ".join(sql_parts)
        
        # Выполнение запроса
        try:
            self.cursor.execute(sql_query, params)
            rows = self.cursor.fetchall()
            
            # Преобразование в SearchResult
            results = []
            for row in rows:
                if row['similarity'] >= similarity_threshold:
                    result = SearchResult(
                        cr_id=row['cr_id'],
                        chunk_index=row['chunk_index'],
                        text=row['text'],
                        similarity=float(row['similarity']),
                        metadata=row['metadata'],
                        cr_title=row.get('cr_title'),
                        icd_codes=row.get('icd_codes')
                    )
                    results.append(result)
            
            logger.info(f"✅ Найдено результатов: {len(results)}")
            if results:
                logger.info(f"   Топ similarity: {results[0].similarity:.4f}")
                logger.info(f"   CR ID: {results[0].cr_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            raise
    
    def format_context(
        self,
        search_results: List[SearchResult],
        max_chars: int = 20000
    ) -> str:
        """
        Форматирование контекста из найденных chunks
        
        Args:
            search_results: Результаты поиска
            max_chars: Максимальное количество символов
            
        Returns:
            Отформатированный контекст для GPT
        """
        if not search_results:
            return "Релевантная информация не найдена."
        
        context_parts = []
        total_chars = 0
        
        # Группировка по CR ID для лучшей структуры
        cr_groups = {}
        for result in search_results:
            if result.cr_id not in cr_groups:
                cr_groups[result.cr_id] = {
                    'title': result.cr_title,
                    'icd_codes': result.icd_codes,
                    'chunks': []
                }
            cr_groups[result.cr_id]['chunks'].append(result)
        
        # Форматирование по группам
        for cr_id, group in cr_groups.items():
            # Заголовок клинической рекомендации
            header = f"\n## {group['title']}\n"
            if group['icd_codes']:
                header += f"Коды МКБ-10: {', '.join(group['icd_codes'])}\n"
            header += f"ID: {cr_id}\n\n"
            
            group_text = header
            
            # Chunks из этой КР
            for chunk in group['chunks']:
                chunk_text = f"{chunk.text}\n\n"
                if total_chars + len(group_text) + len(chunk_text) > max_chars:
                    break
                group_text += chunk_text
            
            if total_chars + len(group_text) > max_chars:
                break
            
            context_parts.append(group_text)
            total_chars += len(group_text)
        
        context = "# КОНТЕКСТ ИЗ КЛИНИЧЕСКИХ РЕКОМЕНДАЦИЙ\n" + "".join(context_parts)
        
        logger.info(f"📝 Контекст сформирован: {len(context)} символов, {len(cr_groups)} КР")
        return context
    
    def generate_answer(
        self,
        query: str,
        context: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Генерация ответа через GPT-4o
        
        Args:
            query: Запрос пользователя
            context: Контекст из базы знаний
            temperature: Температура генерации (0.0 - 1.0)
            max_tokens: Максимальное количество токенов в ответе
            
        Returns:
            Dict с ответом и метаданными
        """
        logger.info(f"\n🤖 Генерация ответа через {self.chat_model}")
        
        # Системный промпт
        system_prompt = """Ты — медицинский ассистент-эксперт, работающий с клиническими рекомендациями Минздрава РФ.

Твоя задача: предоставлять МАКСИМАЛЬНО ДЕТАЛЬНУЮ и ПРАКТИЧЕСКУЮ медицинскую информацию для практикующих врачей на основе клинических рекомендаций.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. НИКОГДА не упоминай слово "контекст" или фразы типа "в контексте", "контекст предоставляет", "согласно контексту"
2. Отвечай УВЕРЕННО, как профессиональный врач-консультант, предоставляя КОНКРЕТНЫЕ детали
3. Используй формулировки: "Согласно клиническим рекомендациям", "В КР указано", "Рекомендуется"
4. ВКЛЮЧАЙ КОНКРЕТНЫЕ ДЕТАЛИ: дозировки препаратов, частоту приема, длительность курса, показатели анализов, критерии диагностики
5. ЦИТИРУЙ точные формулировки из КР, когда это критично (например, критерии диагностики, схемы лечения)
6. НЕ придумывай информацию — используй ТОЛЬКО данные из предоставленного материала
7. НЕ используй форматирование через звездочки ** или другие markdown-символы
8. Используй простой текст с эмодзи для разделения разделов

ОБЯЗАТЕЛЬНАЯ СТРУКТУРА ОТВЕТА (если вопрос о диагнозе/коде МКБ-10):

🩺 Заболевание: [Название]
Коды МКБ-10: [Перечисли ВСЕ упомянутые коды]

🔍 Характерные симптомы:
[Перечисли КОНКРЕТНЫЕ клинические проявления с деталями: частота, тяжесть, особенности проявления. Группируй по категориям если возможно]

🔬 Рекомендуемые обследования и анализы:
[Перечисли ВСЕ упомянутые диагностические процедуры с КОНКРЕТНЫМИ деталями:
- Лабораторные: какие показатели, нормы, интерпретация
- Инструментальные: какие методы, что искать, критерии
- Дифференциальная диагностика: с какими заболеваниями]

💊 Лекарственные назначения:
[Опиши схемы терапии МАКСИМАЛЬНО ПОДРОБНО:
- Препараты первой линии: названия, дозировки, частота приема, длительность
- Альтернативные препараты: когда применять, дозировки
- Комбинированная терапия: схемы сочетания
- Особые указания: противопоказания, взаимодействия, побочные эффекты]

📋 Дополнительные рекомендации:
[ДЕТАЛЬНО укажи:
- Немедикаментозное лечение: диета, режим, физиотерапия
- Показания к госпитализации: конкретные критерии
- Диспансерное наблюдение: сроки, частота, какие обследования
- Прогноз и критерии эффективности лечения]

КРИТИЧЕСКИ ВАЖНО:
- НИКОГДА не пиши "контекст", "в контексте", "контекст предоставляет/не предоставляет"
- Используй фразы: "Согласно КР", "Рекомендуется", "В клинических рекомендациях указано"
- ВКЛЮЧАЙ ВСЕ КОНКРЕТНЫЕ ДЕТАЛИ из клинических рекомендаций: цифры, дозировки, сроки, критерии
- Если упомянуты конкретные исследования или препараты — перечисли их полностью
- Если есть градации (легкая/средняя/тяжелая форма) — опиши различия

Для других типов вопросов (не о диагнозе) — отвечай прямо на вопрос, предоставляя МАКСИМУМ КОНКРЕТНЫХ деталей из КР.

МЕДИЦИНСКИЙ ДИСКЛЕЙМЕР:
Помни, что это информация из клинических рекомендаций для специалистов. 
Окончательные решения принимает лечащий врач с учетом конкретного клинического случая."""

        # Пользовательский промпт
        user_prompt = f"""КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ (извлечено из КР Минздрава РФ):
{context}

---

ВОПРОС ПРАКТИКУЮЩЕГО ВРАЧА:
{query}

ИНСТРУКЦИЯ ДЛЯ ОТВЕТА:
1. В НАЧАЛЕ укажи полное название заболевания и ВСЕ релевантные коды МКБ-10
2. Предоставь МАКСИМАЛЬНО ДЕТАЛЬНУЮ информацию по всем разделам
3. ОБЯЗАТЕЛЬНО включай КОНКРЕТНЫЕ данные:
   - Дозировки препаратов (мг, г, ед.)
   - Частота приема (раз в день, в неделю)
   - Длительность лечения (дни, недели, месяцы)
   - Показатели анализов (референсные значения, критерии)
   - Критерии диагностики (количественные, качественные)
   - Названия конкретных препаратов и схем
4. ЗАПРЕЩЕНО использовать слово "контекст" в любом виде!
5. Используй формулировки: "Согласно КР", "Рекомендуется", "В КР указано"
6. Если в предоставленном материале есть таблицы, схемы, алгоритмы — опиши их содержание подробно
7. Если информации по какому-то разделу нет в предоставленном материале — честно скажи об этом, но НЕ используй слово "контекст"

ВАЖНО: Это консультация для практикующего врача, нужна МАКСИМАЛЬНАЯ конкретика и детализация!"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            
            result = {
                'answer': answer,
                'model': self.chat_model,
                'tokens_prompt': response.usage.prompt_tokens,
                'tokens_completion': response.usage.completion_tokens,
                'tokens_total': response.usage.total_tokens,
                'finish_reason': response.choices[0].finish_reason
            }
            
            logger.info(f"✅ Ответ сгенерирован")
            logger.info(f"   Токенов: {result['tokens_total']} (prompt: {result['tokens_prompt']}, completion: {result['tokens_completion']})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            raise
    
    def _get_cache_key(self, query: str, filter_icd_codes: Optional[List[str]] = None) -> str:
        """Генерация ключа кэша для запроса"""
        cache_input = query.lower().strip()
        if filter_icd_codes:
            cache_input += "|" + "|".join(sorted(filter_icd_codes))
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def ask(
        self,
        query: str,
        n_results: int = 10,
        similarity_threshold: float = 0.5,
        temperature: float = 0.3,
        filter_cr_ids: Optional[List[str]] = None,
        filter_icd_codes: Optional[List[str]] = None,
        user_id: Optional[int] = None,
        username: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Полный цикл RAG: поиск + генерация ответа
        
        Args:
            query: Запрос пользователя
            n_results: Количество chunks для контекста
            similarity_threshold: Минимальный порог similarity
            temperature: Температура генерации GPT
            filter_cr_ids: Фильтр по CR ID
            filter_icd_codes: Фильтр по кодам МКБ-10
            user_id: ID пользователя Telegram
            username: Username пользователя Telegram
            
        Returns:
            Dict с ответом, контекстом, метаданными
        """
        # Проверка кэша для частых запросов (только для простых запросов с кодами МКБ)
        cache_key = None
        if filter_icd_codes and len(filter_icd_codes) == 1:
            cache_key = self._get_cache_key(query, filter_icd_codes)
            if cache_key in self._answer_cache:
                logger.info(f"💾 Ответ взят из кэша (ключ: {cache_key[:8]}...)")
                return self._answer_cache[cache_key]
        
        logger.info("\n" + "="*80)
        logger.info("🚀 RAG PIPELINE START")
        logger.info("="*80)
        
        # 1. Векторный поиск
        search_results = self.search(
            query=query,
            n_results=n_results,
            similarity_threshold=similarity_threshold,
            filter_cr_ids=filter_cr_ids,
            filter_icd_codes=filter_icd_codes
        )
        
        if not search_results:
            return {
                'query': query,
                'answer': "К сожалению, я не нашел релевантной информации в клинических рекомендациях по вашему запросу.",
                'search_results': [],
                'context': "",
                'sources': []
            }
        
        # 2. Форматирование контекста
        context = self.format_context(search_results)
        
        # 3. Генерация ответа
        generation_result = self.generate_answer(
            query=query,
            context=context,
            temperature=temperature
        )
        
        # 4. Извлечение источников (только топ-5 КР по наивысшей similarity)
        # Группируем результаты по CR ID и берем максимальную similarity для каждого
        cr_best_scores = {}
        for result in search_results:
            if result.cr_id not in cr_best_scores or result.similarity > cr_best_scores[result.cr_id]['similarity']:
                cr_best_scores[result.cr_id] = {
                    'cr_id': result.cr_id,
                    'title': result.cr_title,
                    'icd_codes': result.icd_codes,
                    'similarity': result.similarity,
                    'url': f"https://cr.minzdrav.gov.ru/view-cr/{result.cr_id}"
                }
        
        # Сортируем по similarity и берем топ-5
        sources = sorted(cr_best_scores.values(), key=lambda x: x['similarity'], reverse=True)[:5]
        
        # 5. Итоговый результат
        final_result = {
            'query': query,
            'answer': generation_result['answer'],
            'search_results': [
                {
                    'cr_id': r.cr_id,
                    'chunk_index': r.chunk_index,
                    'text': r.text[:200] + '...' if len(r.text) > 200 else r.text,
                    'similarity': r.similarity,
                    'cr_title': r.cr_title
                }
                for r in search_results[:5]  # Топ-5 для вывода
            ],
            'context': context,
            'sources': sources,
            'tokens': {
                'prompt': generation_result['tokens_prompt'],
                'completion': generation_result['tokens_completion'],
                'total': generation_result['tokens_total']
            }
        }
        
        logger.info("\n" + "="*80)
        logger.info("✅ RAG PIPELINE COMPLETE")
        logger.info("="*80)
        
        # Расчет стоимости запроса
        # Цены OpenAI (ноябрь 2024)
        # text-embedding-3-large: $0.13 / 1M tokens
        # gpt-4o: $2.50 / 1M input tokens, $10.00 / 1M output tokens
        embedding_cost = 0.13 / 1_000_000  # per token
        gpt_input_cost = 2.50 / 1_000_000  # per token
        gpt_output_cost = 10.00 / 1_000_000  # per token
        
        # Примерный размер embedding запроса (query обычно ~20-100 токенов)
        query_tokens_estimate = len(query.split()) * 1.3  # грубая оценка
        
        total_cost = (
            query_tokens_estimate * embedding_cost +  # embedding
            generation_result['tokens_prompt'] * gpt_input_cost +  # GPT input
            generation_result['tokens_completion'] * gpt_output_cost  # GPT output
        )
        
        # Логирование стоимости и деталей запроса
        self._log_query_cost(
            query=query,
            answer=generation_result['answer'],
            tokens_embedding=query_tokens_estimate,
            tokens_prompt=generation_result['tokens_prompt'],
            tokens_completion=generation_result['tokens_completion'],
            cost=total_cost,
            sources=sources,
            user_id=user_id,
            username=username,
            cached=False,
            n_results=n_results,
            similarity_threshold=similarity_threshold,
            top_similarity=search_results[0].similarity if search_results else 0.0,
            results_count=len(search_results),
            context_chars=len(context),
            context_crs=len(set(r.cr_id for r in search_results))
        )
        
        # Сохранение в кэш (только для запросов с 1 кодом МКБ)
        if cache_key:
            if len(self._answer_cache) >= self._cache_max_size:
                # Удаляем самый старый элемент (FIFO)
                self._answer_cache.pop(next(iter(self._answer_cache)))
            self._answer_cache[cache_key] = final_result
            logger.info(f"💾 Ответ сохранен в кэш (всего в кэше: {len(self._answer_cache)})")
        
        return final_result
    
    def _log_query_cost(
        self,
        query: str,
        answer: str,
        tokens_embedding: float,
        tokens_prompt: int,
        tokens_completion: int,
        cost: float,
        sources: list,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        cached: bool = False,
        n_results: int = 0,
        similarity_threshold: float = 0.0,
        top_similarity: float = 0.0,
        results_count: int = 0,
        context_chars: int = 0,
        context_crs: int = 0,
        execution_time_ms: int = 0
    ):
        """
        Логирование стоимости и деталей запроса в БД
        
        Args:
            query: Запрос пользователя
            answer: Сгенерированный ответ
            tokens_embedding: Токены embedding
            tokens_prompt: Токены промпта GPT
            tokens_completion: Токены ответа GPT
            cost: Общая стоимость в USD
            sources: Список источников
            user_id: ID пользователя Telegram
            username: Username пользователя
            cached: Был ли ответ из кэша
            n_results: Количество запрошенных результатов
            similarity_threshold: Порог similarity
            top_similarity: Топ similarity из результатов
            results_count: Количество найденных результатов
            context_chars: Размер контекста в символах
            context_crs: Количество КР в контексте
            execution_time_ms: Время выполнения в мс
        """
        # Цены OpenAI
        embedding_cost_rate = 0.13 / 1_000_000
        gpt_input_cost_rate = 2.50 / 1_000_000
        gpt_output_cost_rate = 10.00 / 1_000_000
        
        embedding_cost = tokens_embedding * embedding_cost_rate
        input_cost = tokens_prompt * gpt_input_cost_rate
        output_cost = tokens_completion * gpt_output_cost_rate
        
        try:
            # Запись в БД
            sql = """
                INSERT INTO query_logs (
                    query, user_id, username, n_results, similarity_threshold,
                    top_similarity, results_count, context_chars, context_crs,
                    prompt_tokens, completion_tokens, total_tokens,
                    embedding_cost, input_cost, output_cost, total_cost,
                    cached, response_length, execution_time_ms
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
            """
            
            self.cursor.execute(sql, (
                query, user_id, username, n_results, similarity_threshold,
                top_similarity, results_count, context_chars, context_crs,
                tokens_prompt, tokens_completion, 
                tokens_prompt + tokens_completion,
                embedding_cost, input_cost, output_cost, cost,
                cached, len(answer), execution_time_ms
            ))
            self.conn.commit()
            
            logger.info(f"💰 Стоимость запроса: ${cost:.6f}")
            logger.info(f"   Токены: embedding={tokens_embedding:.0f}, input={tokens_prompt}, output={tokens_completion}")
            logger.info(f"   Лог сохранен в БД (ID: {self.cursor.lastrowid if hasattr(self.cursor, 'lastrowid') else 'N/A'})")
            
        except Exception as e:
            logger.error(f"❌ Ошибка логирования в БД: {e}")
            # Fallback: записываем в файл если БД не доступна
            self._log_to_file(query, answer, tokens_embedding, tokens_prompt, 
                            tokens_completion, cost, sources)
    
    def _log_to_file(
        self,
        query: str,
        answer: str,
        tokens_embedding: float,
        tokens_prompt: int,
        tokens_completion: int,
        cost: float,
        sources: list
    ):
        """Fallback: логирование в файл если БД недоступна"""
        import json
        from datetime import datetime
        
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
            "answer_full_length": len(answer),
            "tokens": {
                "embedding": round(tokens_embedding, 2),
                "gpt_input": tokens_prompt,
                "gpt_output": tokens_completion,
                "total": round(tokens_embedding + tokens_prompt + tokens_completion, 2)
            },
            "cost_usd": round(cost, 6),
            "sources_count": len(sources),
            "sources": [
                {
                    "title": src['title'],
                    "cr_id": src['cr_id'],
                    "similarity": round(src['similarity'], 4)
                }
                for src in sources
            ]
        }
        
        log_file = os.path.join(log_dir, f"query_costs_{datetime.now().strftime('%Y-%m-%d')}.jsonl")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        logger.info(f"   Fallback: лог сохранен в файл {log_file}")


def main():
    """Тестовый запуск RAG системы"""
    
    # Примеры запросов
    test_queries = [
        "Какие методы диагностики миокардита?",
        "Лечение инфаркта миокарда",
        "Показания к госпитализации при эпилепсии"
    ]
    
    with RAGSystem() as rag:
        for query in test_queries:
            print("\n" + "="*80)
            print(f"ЗАПРОС: {query}")
            print("="*80)
            
            result = rag.ask(
                query=query,
                n_results=5,
                similarity_threshold=0.5
            )
            
            print(f"\nОТВЕТ:\n{result['answer']}\n")
            print(f"\nИСТОЧНИКИ:")
            for source in result['sources']:
                print(f"  - {source['title']} (МКБ-10: {', '.join(source['icd_codes'])})")
            
            print(f"\nТОКЕНОВ: {result['tokens']['total']}")
            print("="*80)


if __name__ == "__main__":
    main()
