import requests
from bs4 import BeautifulSoup
import re
import json
import time
import pandas as pd
from playwright.sync_api import sync_playwright
import os
from typing import List, Dict, Any
import csv


class SemanticSegmenter:
    def __init__(self):
        self.MIN_CHUNK_SIZE = 400
        self.MAX_CHUNK_SIZE = 1800

        # Списки для извлечения (I00-I99)
        self.examinations = [
            'ЭКГ', 'ЭХОКГ', 'ХМЭКГ', 'МРТ', 'КТ', 'УЗИ', 'ВСЭФИ', 'ЧПЭФИ',
            'ЭФИ', 'КАГ', 'ЧКВ', 'Сцинтиграфия', 'Коронарография',
            'Вентрикулография', 'Ангиография', 'Велоэргометрия', 'Тредмил',
            'Стресс-тест', 'Нагрузочная проба', 'Стресс-ЭХОКГ', 'СМАД',
            'Холтер', 'Ритмография', 'анализ крови', 'биохимический анализ',
            'липидограмма', 'коагулограмма', 'тропонин', 'МВ-КФК', 'BNP',
            'NT-proBNP', 'С-реактивный белок', 'гомоцистеин', 'Эхокардиография',
            'Допплерография', 'ПЭТ', 'ТЭЭ', 'Суточное мониторирование', 'Кардиориск'
        ]

        self.symptoms = [
            'сердцебиение', 'перебои', 'аритмия', 'тахикардия', 'брадикардия',
            'боль в груди', 'стенокардия', 'давящая боль', 'жжение в груди',
            'ощущение остановки сердца', 'замирание сердца', 'одышка', 'удушье',
            'нехватка воздуха', 'затрудненное дыхание', 'ортопноэ', 'ночное апноэ',
            'головокружение', 'обморок', 'синкопе', 'предобморочное состояние',
            'головная боль', 'шум в ушах', 'мелькание мушек', 'слабость',
            'утомляемость', 'снижение толерантности к нагрузке',
            'непереносимость физической нагрузки', 'отеки', 'отек ног',
            'пастозность', 'набухание вен', 'тяжесть в правом подреберье',
            'увеличение живота', 'цианоз', 'посинение', 'бледность',
            'холодные конечности', 'акроцианоз', 'тошнота', 'рвота',
            'потеря аппетита', 'вздутие живота', 'ночные пробуждения',
            'кашель ночной', 'сердцебиение ночью', 'тревога', 'страх',
            'паника', 'ощущение страха', 'потливость', 'холодный пот',
            'артралгия', 'миалгия'
        ]

    def extract_metadata(self, text: str) -> Dict[str, List]:
        """Извлекает метаданные из текста"""
        # МКБ коды
        mkb_codes = re.findall(r'[I][0-9]{2}(?:\.[0-9])?', text.upper())

        # Лекарства (между **)
        medications = re.findall(r'\*\*(.*?)\*\*', text)

        # Обследования
        found_examinations = []
        for exam in self.examinations:
            if re.search(r'\b' + re.escape(exam) + r'\b', text, re.IGNORECASE):
                found_examinations.append(exam)

        # Симптомы
        found_symptoms = []
        for symptom in self.symptoms:
            if re.search(r'\b' + re.escape(symptom) + r'\b', text, re.IGNORECASE):
                found_symptoms.append(symptom)

        return {
            'mkb_codes': list(set(mkb_codes)),
            'medications': list(set(medications)),
            'examinations': list(set(found_examinations)),
            'symptoms': list(set(found_symptoms))
        }

    def semantic_segmentation(self, html_content: str, document_id: str) -> List[Dict]:
        """Основная функция семантической сегментации"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Сначала получаем все заголовки и их уровни
        headings = self._extract_headings_with_hierarchy(soup)

        # Строим дерево разделов
        sections_tree = self._build_sections_tree(soup, headings)

        # Создаем чанки из дерева
        chunks = self._create_chunks_from_tree(sections_tree, document_id)

        return chunks

    def _extract_headings_with_hierarchy(self, soup) -> List[Dict]:
        """Извлекает заголовки с информацией об иерархии"""
        headings = []

        # Ищем все возможные заголовки
        for level in range(1, 7):
            elements = soup.find_all(f'h{level}')
            for element in elements:
                text = element.get_text().strip()
                if text and len(text) > 5:
                    # Пытаемся определить номер раздела (1.1.1 и т.д.)
                    section_number = self._extract_section_number(text)

                    headings.append({
                        'element': element,
                        'level': level,
                        'text': text,
                        'section_number': section_number,
                        'id': element.get('id', '')
                    })

        return headings

    def _extract_section_number(self, text: str) -> str:
        """Извлекает номер раздела из текста заголовка"""
        # Ищем паттерны типа "1.1", "2.3.1" и т.д.
        match = re.match(r'^(\d+(?:\.\d+)*)\.', text)
        return match.group(1) if match else ""

    def _build_sections_tree(self, soup, headings: List[Dict]) -> Dict:
        """Строит дерево разделов документа"""
        # Сортируем заголовки по положению в документе
        headings.sort(key=lambda x: self._get_element_position(x['element']))

        root = {
            'type': 'root',
            'children': [],
            'full_hierarchy': [],
            'level': 0
        }

        current_path = [root]

        for heading in headings:
            # Определяем уровень вложенности по номеру раздела
            level = heading['section_number'].count('.') + 1 if heading['section_number'] else heading['level']

            # Находим правильного родителя
            while len(current_path) > level:
                current_path.pop()

            parent = current_path[-1]

            # Собираем контент до следующего заголовка
            content = self._get_content_until_next_heading(heading['element'])

            new_section = {
                'type': 'section',
                'header': heading['text'],
                'section_number': heading['section_number'],
                'content': content,
                'level': level,
                'full_hierarchy': parent['full_hierarchy'] + [heading['text']],
                'children': []
            }

            parent['children'].append(new_section)
            current_path.append(new_section)

        return root

    def _get_content_until_next_heading(self, heading_element) -> str:
        """Собирает контент от заголовка до следующего заголовка"""
        content_parts = []
        current = heading_element.next_sibling

        while current and not (hasattr(current, 'name') and current.name and current.name.startswith('h')):
            if hasattr(current, 'get_text'):
                text = current.get_text().strip()
                if text:
                    content_parts.append(text)
            elif hasattr(current, 'string') and current.string:
                text = current.string.strip()
                if text:
                    content_parts.append(text)
            current = current.next_sibling

        return ' '.join(content_parts)

    def _create_chunks_from_tree(self, tree: Dict, document_id: str) -> List[Dict]:
        """Создает чанки из дерева разделов"""
        chunks = []

        def process_node(node, chunk_id_prefix=""):
            if node['type'] == 'section':
                content = node['content']

                # Проверяем размер контента
                if len(content) < self.MIN_CHUNK_SIZE and node['children']:
                    # Объединяем с первым ребенком
                    if node['children']:
                        first_child = node['children'][0]
                        combined_content = content + " " + first_child['content']
                        combined_hierarchy = node['full_hierarchy']

                        # Создаем объединенный чанк
                        if len(combined_content) <= self.MAX_CHUNK_SIZE:
                            metadata = self.extract_metadata(combined_content)
                            chunk = {
                                'chunk_id': f"{document_id}_{chunk_id_prefix}",
                                'header': ' -> '.join(combined_hierarchy),
                                'full_hierarchy': combined_hierarchy,
                                'content': combined_content,
                                'metadata': metadata
                            }
                            chunks.append(chunk)

                            # Пропускаем первого ребенка (уже объединен)
                            remaining_children = node['children'][1:]
                            for i, child in enumerate(remaining_children):
                                process_node(child, f"{chunk_id_prefix}.{i + 2}")
                            return

                # Обычная обработка
                if len(content) <= self.MAX_CHUNK_SIZE:
                    # Создаем один чанк
                    metadata = self.extract_metadata(content)
                    chunk = {
                        'chunk_id': f"{document_id}_{chunk_id_prefix}",
                        'header': ' -> '.join(node['full_hierarchy']),
                        'full_hierarchy': node['full_hierarchy'],
                        'content': content,
                        'metadata': metadata
                    }
                    chunks.append(chunk)
                else:
                    # Делим большой контент
                    sub_chunks = self._split_large_content(content, node['full_hierarchy'], document_id,
                                                           chunk_id_prefix)
                    chunks.extend(sub_chunks)

                # Обрабатываем детей
                for i, child in enumerate(node['children']):
                    process_node(child, f"{chunk_id_prefix}.{i + 1}")

        # Запускаем обработку с корня
        for i, child in enumerate(tree['children']):
            process_node(child, str(i + 1))

        return chunks

    def _split_large_content(self, content: str, hierarchy: List[str],
                             document_id: str, base_chunk_id: str) -> List[Dict]:
        """Делит большой контент на части"""
        chunks = []

        # Делим по абзацам или предложениям
        paragraphs = re.split(r'\n\s*\n', content)
        current_chunk = ""
        chunk_index = 1

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) <= self.MAX_CHUNK_SIZE:
                current_chunk += " " + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    # Сохраняем текущий чанк
                    metadata = self.extract_metadata(current_chunk)
                    chunk = {
                        'chunk_id': f"{document_id}_{base_chunk_id}.{chunk_index}",
                        'header': ' -> '.join(hierarchy) + f" [Часть {chunk_index}]",
                        'full_hierarchy': hierarchy + [f"Часть {chunk_index}"],
                        'content': current_chunk,
                        'metadata': metadata
                    }
                    chunks.append(chunk)
                    chunk_index += 1

                current_chunk = paragraph

        # Добавляем последний чанк
        if current_chunk:
            metadata = self.extract_metadata(current_chunk)
            chunk = {
                'chunk_id': f"{document_id}_{base_chunk_id}.{chunk_index}",
                'header': ' -> '.join(hierarchy) + f" [Часть {chunk_index}]",
                'full_hierarchy': hierarchy + [f"Часть {chunk_index}"],
                'content': current_chunk,
                'metadata': metadata
            }
            chunks.append(chunk)

        return chunks

    def _get_element_position(self, element) -> int:
        """Вспомогательная функция для определения позиции элемента"""
        # Простая реализация
        return str(element).find(str(element))


class MinZdravCrParser:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.base_url = "https://cr.minzdrav.gov.ru"
        self.segmenter = SemanticSegmenter()

    def get_document_ids_from_local_excel(self, excel_path: str) -> List[str]:
        """
        Получает ID рекомендаций из локального Excel файла
        Фильтрует по МКБ-10 (I00-I99, G00-G99) и статусу 'Применяется'
        """
        print("📁 ЗАГРУЗКА ID ИЗ ЛОКАЛЬНОГО EXCEL ФАЙЛА")
        print("-" * 50)

        try:
            # Пробуем загрузить Excel с разными движками
            try:
                df = pd.read_excel(excel_path, engine='openpyxl')
            except ImportError:
                try:
                    df = pd.read_excel(excel_path, engine='xlrd')
                except ImportError:
                    print("❌ Не установлены библиотеки для чтения Excel. Установите: pip install openpyxl")
                    return self._get_test_ids()

            print(f"✅ Excel загружен, строк: {len(df)}")
            print(f"📊 Колонки: {list(df.columns)}")

            # Определяем названия колонок (на основе вашего примера)
            id_col = None
            mkb_col = None
            status_col = None

            for col in df.columns:
                # Ищем колонку с ID
                if id_col is None and any(df[col].astype(str).str.match(r'\d+_\d+').any() for _ in df[col].head(3)):
                    id_col = col
                # Ищем колонку с МКБ-10
                if mkb_col is None and any(
                        'I' in str(val) or 'G' in str(val) for val in df[col].head(3) if pd.notna(val)):
                    mkb_col = col
                # Ищем колонку со статусом
                if status_col is None and any('Применяется' in str(val) for val in df[col].head(3) if pd.notna(val)):
                    status_col = col

            print(f"🔍 Найдены колонки: ID='{id_col}', МКБ-10='{mkb_col}', Статус='{status_col}'")

            if not all([id_col, mkb_col, status_col]):
                print("❌ Не удалось найти все необходимые колонки")
                return self._get_test_ids()

            # Фильтруем данные
            filtered_ids = []

            for idx, row in df.iterrows():
                try:
                    document_id = str(row[id_col])
                    mkb_codes = str(row[mkb_col]) if pd.notna(row[mkb_col]) else ""
                    status = str(row[status_col]) if pd.notna(row[status_col]) else ""

                    # Проверяем условия
                    if (status == "Применяется" and
                            any(code.strip().startswith(('I', 'G'))
                                for code in mkb_codes.split(',') if code.strip())):
                        filtered_ids.append(document_id)
                        print(f"✅ {document_id}: {mkb_codes}")

                except Exception as e:
                    print(f"❌ Ошибка обработки строки {idx}: {e}")
                    continue

            print(f"🎯 Найдено подходящих документов: {len(filtered_ids)}")
            return filtered_ids

        except Exception as e:
            print(f"❌ Ошибка чтения Excel: {e}")
            print("🔄 Пробуем прочитать как CSV...")
            return self._try_read_as_csv(excel_path)

    def _try_read_as_csv(self, file_path: str) -> List[str]:
        """Пытается прочитать файл как CSV"""
        try:
            # Пробуем разные разделители
            for delimiter in [',', ';', '\t']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                    print(f"✅ CSV загружен с разделителем '{delimiter}', строк: {len(df)}")
                    return self._filter_csv_data(df)
                except:
                    continue

            # Если UTF-8 не работает, пробуем другие кодировки
            for encoding in ['cp1251', 'windows-1251', 'latin1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ CSV загружен с кодировкой {encoding}, строк: {len(df)}")
                    return self._filter_csv_data(df)
                except:
                    continue

            print("❌ Не удалось прочитать файл как CSV")
            return self._get_test_ids()

        except Exception as e:
            print(f"❌ Ошибка чтения CSV: {e}")
            return self._get_test_ids()

    def _filter_csv_data(self, df: pd.DataFrame) -> List[str]:
        """Фильтрует данные из DataFrame"""
        # Определяем колонки (аналогично Excel методу)
        id_col = None
        mkb_col = None
        status_col = None

        for col in df.columns:
            # Ищем колонку с ID
            if id_col is None and any(df[col].astype(str).str.match(r'\d+_\d+').any() for _ in df[col].head(3)):
                id_col = col
            # Ищем колонку с МКБ-10
            if mkb_col is None and any('I' in str(val) or 'G' in str(val) for val in df[col].head(3) if pd.notna(val)):
                mkb_col = col
            # Ищем колонку со статусом
            if status_col is None and any('Применяется' in str(val) for val in df[col].head(3) if pd.notna(val)):
                status_col = col

        if not all([id_col, mkb_col, status_col]):
            print("❌ Не удалось найти все необходимые колонки в CSV")
            return self._get_test_ids()

        filtered_ids = []

        for idx, row in df.iterrows():
            try:
                document_id = str(row[id_col])
                mkb_codes = str(row[mkb_col]) if pd.notna(row[mkb_col]) else ""
                status = str(row[status_col]) if pd.notna(row[status_col]) else ""

                # Проверяем условия
                if (status == "Применяется" and
                        any(code.strip().startswith(('I', 'G'))
                            for code in mkb_codes.split(',') if code.strip())):
                    filtered_ids.append(document_id)
                    print(f"✅ {document_id}: {mkb_codes}")

            except Exception as e:
                print(f"❌ Ошибка обработки строки {idx}: {e}")
                continue

        print(f"🎯 Найдено подходящих документов в CSV: {len(filtered_ids)}")
        return filtered_ids

    def _get_test_ids(self):
        """Тестовые ID для работы"""
        test_ids = [
            '539_2',  # Анемии при злокачественных новообразованиях
            '960_1',  # Тромбоз глубоких вен конечностей (I80...)
            '961_1',  # Гипертрофическая кардиомиопатия у детей (I42...)
        ]
        print(f"✅ Используем тестовые ID: {test_ids}")
        return test_ids

    def get_document_html(self, document_id):
        """Получает HTML документа"""
        url = f"{self.base_url}/view-cr/{document_id}"
        print(f"🌐 Загрузка: {document_id}")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_default_timeout(60000)

                print("   Загрузка через Playwright...")
                page.goto(url, wait_until="networkidle")

                # Ждем загрузки контента
                page.wait_for_timeout(5000)

                # Проверяем основные элементы
                content = page.content()

                # Сохраняем для отладки
                with open(f"debug_{document_id}.html", "w", encoding="utf-8") as f:
                    f.write(content)

                browser.close()

                print(f"✅ Документ {document_id} загружен")
                return content

        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return None

    def parse_document_semantic(self, html_content, document_id):
        """Парсит документ с семантической сегментацией"""
        if not html_content:
            return None

        print(f"\n🔍 СЕМАНТИЧЕСКАЯ СЕГМЕНТАЦИЯ ДОКУМЕНТА {document_id}")
        print("=" * 60)

        try:
            chunks = self.segmenter.semantic_segmentation(html_content, document_id)

            # Извлекаем общее название документа
            soup = BeautifulSoup(html_content, 'html.parser')
            title = self._extract_title(soup)

            print(f"✅ Создано {len(chunks)} чанков")

            # Сохраняем результат
            result = {
                'document_id': document_id,
                'title': title,
                'total_chunks': len(chunks),
                'chunks': chunks
            }

            return result

        except Exception as e:
            print(f"❌ Ошибка семантической сегментации: {e}")
            return None

    def _extract_title(self, soup):
        """Извлекает название"""
        selectors = ['.title-content', 'h1', '.document-title', 'title']

        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                if title and len(title) > 10:
                    print(f"📖 Название: {title[:80]}...")
                    return title

        return "Неизвестно"

    def process_documents_with_semantic_segmentation(self, excel_path: str, limit: int = 5):
        """
        Основная функция обработки документов с семантической сегментацией
        """
        print("🚀 ЗАПУСК ОБРАБОТКИ С СЕМАНТИЧЕСКОЙ СЕГМЕНТАЦИЕЙ")
        print("=" * 60)

        # Получаем ID из локального Excel
        document_ids = self.get_document_ids_from_local_excel(excel_path)

        if not document_ids:
            print("❌ Не найдено подходящих документов")
            return []

        print(f"📋 Всего найдено документов: {len(document_ids)}")
        print(f"🔧 Обрабатываем первые {limit} документов")

        results = []
        processed_count = 0

        for i, doc_id in enumerate(document_ids[:limit]):
            print(f"\n{'=' * 60}")
            print(f"📄 ДОКУМЕНТ {i + 1}/{min(limit, len(document_ids))}: {doc_id}")
            print('=' * 60)

            # Загружаем HTML
            html = self.get_document_html(doc_id)
            if html:
                # Обрабатываем с семантической сегментацией
                result = self.parse_document_semantic(html, doc_id)
                if result:
                    results.append(result)
                    processed_count += 1

                    # Сохраняем результат
                    filename = f"semantic_chunks_{doc_id}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"💾 Сохранено: {filename}")

                    # Показываем статистику по чанкам
                    total_chunks = result['total_chunks']
                    chunks_with_meds = sum(1 for chunk in result['chunks'] if chunk['metadata']['medications'])
                    chunks_with_exams = sum(1 for chunk in result['chunks'] if chunk['metadata']['examinations'])
                    chunks_with_symptoms = sum(1 for chunk in result['chunks'] if chunk['metadata']['symptoms'])

                    print(f"📊 Статистика чанков:")
                    print(f"   • Всего чанков: {total_chunks}")
                    print(f"   • С лекарствами: {chunks_with_meds}")
                    print(f"   • С обследованиями: {chunks_with_exams}")
                    print(f"   • С симптомами: {chunks_with_symptoms}")

            time.sleep(2)  # Пауза между запросами

        print(f"\n✅ ГОТОВО! Обработано: {processed_count} документов")
        return results


def main():
    """ОСНОВНАЯ ФУНКЦИЯ"""
    print("🚀 ПАРСЕР КЛИНИЧЕСКИХ РЕКОМЕНДАЦИЙ МИНЗДРАВА")
    print("С СЕМАНТИЧЕСКОЙ СЕГМЕНТАЦИЕЙ")
    print("=" * 60)

    parser = MinZdravCrParser()

    # Путь к вашему Excel файлу
    excel_path = r"D:\загрузки с яндекса\Список утвержденных клинических рекомендаций.xlsx"

    # Проверяем существование файла
    if not os.path.exists(excel_path):
        print(f"❌ Файл не найден: {excel_path}")
        print("🔄 Используем тестовые ID...")
        excel_path = None
    else:
        print(f"✅ Найден файл: {excel_path}")

    # Обрабатываем документы
    results = parser.process_documents_with_semantic_segmentation(
        excel_path=excel_path,
        limit=100  # Можно увеличить
    )

    # Сводка
    if results:
        print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        total_chunks = sum(result['total_chunks'] for result in results)
        print(f"   • Обработано документов: {len(results)}")
        print(f"   • Всего создано чанков: {total_chunks}")

        for result in results:
            print(f"   • {result['document_id']}: {result['title'][:50]}...")
            print(f"     Чанков: {result['total_chunks']}")


if __name__ == "__main__":
    main()