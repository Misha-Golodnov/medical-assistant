"""
Просмотр логов запросов из PostgreSQL
"""
import os
import sys
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Подключение к PostgreSQL"""
    return psycopg2.connect(
        os.getenv('DATABASE_URL'),
        cursor_factory=RealDictCursor
    )


def view_logs(limit=20, hours=24):
    """
    Просмотр последних N запросов за указанное количество часов
    
    Args:
        limit: Максимальное количество записей
        hours: Количество часов назад от текущего момента
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    time_filter = datetime.now() - timedelta(hours=hours)
    
    cursor.execute("""
        SELECT 
            id,
            query,
            user_id,
            username,
            n_results,
            similarity_threshold,
            top_similarity,
            results_count,
            context_chars,
            context_crs,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            embedding_cost,
            input_cost,
            output_cost,
            total_cost,
            cached,
            response_length,
            execution_time_ms,
            error,
            created_at
        FROM query_logs
        WHERE created_at >= %s
        ORDER BY created_at DESC
        LIMIT %s
    """, (time_filter, limit))
    
    logs = cursor.fetchall()
    
    if not logs:
        print(f"\n❌ Нет записей за последние {hours} часов")
        conn.close()
        return
    
    print(f"\n📊 Последние {len(logs)} запросов за {hours}ч\n")
    print("=" * 100)
    
    total_cost = 0
    total_tokens = 0
    
    for log in logs:
        print(f"\n🔍 ID: {log['id']} | {log['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Запрос: {log['query'][:80]}...")
        
        if log['user_id']:
            print(f"   Пользователь: {log['username']} (ID: {log['user_id']})")
        
        print(f"   Параметры: n_results={log['n_results']}, threshold={log['similarity_threshold']}")
        
        if log['error']:
            print(f"   ❌ Ошибка: {log['error']}")
        else:
            print(f"   Результаты: {log['results_count']} chunks, top_similarity={log['top_similarity']:.3f}")
            print(f"   Контекст: {log['context_chars']} символов, {log['context_crs']} КР")
            print(f"   Токены: {log['prompt_tokens']} input + {log['completion_tokens']} output = {log['total_tokens']} total")
            print(f"   💰 Стоимость: ${log['total_cost']:.6f} (embed: ${log['embedding_cost']:.6f}, in: ${log['input_cost']:.6f}, out: ${log['output_cost']:.6f})")
            print(f"   Ответ: {log['response_length']} символов за {log['execution_time_ms']}ms")
            
            if log['cached']:
                print(f"   ⚡ Кэшировано")
            
            total_cost += float(log['total_cost'])
            total_tokens += log['total_tokens']
    
    print("\n" + "=" * 100)
    print(f"\n📈 Итого: {len(logs)} запросов, {total_tokens} токенов, ${total_cost:.6f}")
    
    conn.close()


def view_stats():
    """Статистика по пользователям"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            user_id,
            username,
            COUNT(*) as query_count,
            SUM(total_tokens) as total_tokens,
            SUM(total_cost) as total_cost,
            AVG(total_cost) as avg_cost,
            MAX(created_at) as last_query
        FROM query_logs
        WHERE error IS NULL
        GROUP BY user_id, username
        ORDER BY query_count DESC
    """)
    
    stats = cursor.fetchall()
    
    if not stats:
        print("\n❌ Нет данных для статистики")
        conn.close()
        return
    
    print(f"\n📊 Статистика по пользователям\n")
    print("=" * 100)
    
    for stat in stats:
        user_label = f"{stat['username']} (ID: {stat['user_id']})" if stat['user_id'] else "Анонимный"
        print(f"\n👤 {user_label}")
        print(f"   Запросов: {stat['query_count']}")
        print(f"   Токенов: {stat['total_tokens']}")
        print(f"   💰 Общая стоимость: ${float(stat['total_cost']):.6f}")
        print(f"   💰 Средняя стоимость: ${float(stat['avg_cost']):.6f}")
        print(f"   Последний запрос: {stat['last_query'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 100)
    
    # Общая статистика
    cursor.execute("""
        SELECT 
            COUNT(*) as total_queries,
            SUM(total_tokens) as total_tokens,
            SUM(total_cost) as total_cost,
            AVG(execution_time_ms) as avg_time_ms,
            COUNT(CASE WHEN cached THEN 1 END) as cached_count
        FROM query_logs
        WHERE error IS NULL
    """)
    
    overall = cursor.fetchone()
    
    print(f"\n🌍 Общая статистика:")
    print(f"   Всего запросов: {overall['total_queries']}")
    print(f"   Кэшировано: {overall['cached_count']} ({overall['cached_count']/overall['total_queries']*100:.1f}%)")
    print(f"   Токенов: {overall['total_tokens']}")
    print(f"   💰 Общая стоимость: ${float(overall['total_cost']):.6f}")
    print(f"   ⏱️  Среднее время: {overall['avg_time_ms']:.0f}ms")
    
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "stats":
            view_stats()
        elif arg == "all":
            view_logs(limit=1000, hours=24*7)  # Все за неделю
        else:
            try:
                limit = int(arg)
                view_logs(limit=limit)
            except ValueError:
                print("❌ Использование: python view_logs.py [limit|stats|all]")
                print("   limit - число записей (по умолчанию 20)")
                print("   stats - статистика по пользователям")
                print("   all - все записи за неделю")
    else:
        view_logs()
