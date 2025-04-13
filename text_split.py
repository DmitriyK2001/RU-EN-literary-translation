import os
import re
import sys


def split_book(file_path, output_dir, delimiter_pattern, folder_number, lang):
    # Читаем весь текст книги
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    REMOVE_PAGE_MARKERS = False
    if REMOVE_PAGE_MARKERS:
        # Регулярное выражение, которое ищет вхождения вида "p. 4", "p.5" и т.п.
        # \b обеспечивает, что "p." является отдельным словом (не удалятся окончания типа "up.", "stop.")
        page_pattern = r'\bp\.\s*\d+\b'
        text = re.sub(page_pattern, '', text)
    
    # Разбиваем текст по шаблону разделителя.
    # Флаг re.IGNORECASE позволяет не учитывать регистр, а re.MULTILINE – обрабатывать каждую строку отдельно.
    chapters = re.split(delimiter_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Иногда до первой главы может идти предисловие – можно его пропустить, если нужно.
    chapters = [ch.strip() for ch in chapters if ch.strip()]

    # Создаём выходную папку, если её ещё нет
    os.makedirs(output_dir, exist_ok=True)

    # Записываем каждую главу в отдельный файл
    for i, chapter in enumerate(chapters, start=1):
        filename = os.path.join(output_dir, f"{folder_number}_{i}_{lang}.txt")
        with open(filename, 'w', encoding='utf-8') as out_file:
            out_file.write(chapter)
        print(f"Глава {i} сохранена в файле: {filename}")

if __name__ == "__main__":

    current_folder = os.path.basename(os.getcwd())
    m = re.search(r'\d+', current_folder)
    folder_number = m.group(0) if m else current_folder
    input_file_one = f"{folder_number}_copy_en.txt"
    input_file_two = f"{folder_number}_copy_ru.txt"
    output_dir = "./"

    # Пример шаблона для разделителя:
    # Он ищет строки, начинающиеся с "chapter" и за ними римские цифры.
    # При необходимости можно задать другой шаблон, например, для строк, состоящих только из римских цифр.
    #delimiter_pattern = r'(?=^chapter\s+[ivxlcdm1234567890]+\b)'
    #delimiter_pattern = r'(?=^\s*(?:глава|chapter)\s+(?:[ivxlcdmхс]+|[0-9]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\b)'
    #delimiter_pattern = r'(?=^\s*(?:(?:глава|chapter)\s+(?:[ivxlcdmхс]+|[0-9]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty)\b|(?:[ivxlcdm]+\.)(?:\s*$)?))'
    delimiter_pattern_en = r'(?=^\s*chapter\s+(?:[ivxlcdm]+|[0-9]+|the|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b)'
    delimiter_pattern_ru = r'(?=^\s*глава|часть\s+(?:[ivxlcdmхс]+|[0-9]+|первая|вторая|третья|четвертая|пятая|шестая|седьмая|восьмая|девятая|десятая|одиннадцатая|двенадцатая|тринадцатая|четырнадцатая|пятнадцатая|шестнадцатая|семнадцатая|восемнадцатая|девятнадцатая|двадцатая|двадцать|тридцать|сорок|пятьдесят|шестьдесят|семьдесят|восемьдесят|девяносто|сто)\b)'
    delimiter_pattern_alt = r'(?=^(?:[ivxlcdm]+|[0-9]+)\.)'
    #delimiter_pattern = r'(?=^\s*глава\s+[ivxlcdmхс1234567890]+\b)'

    split_book(input_file_one, output_dir, delimiter_pattern_en, folder_number, 'en')
    split_book(input_file_two, output_dir, delimiter_pattern_ru, folder_number, 'ru')