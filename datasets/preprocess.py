import json

def read_chapter(file_path):
    # Читаем весь файл и объединяем строки в одну длинную строку
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Убираем лишние пробелы и переводы строк
    long_text = " ".join(text.split())
    return long_text

dataset = []
# Пути к файлам с главами
n_text = [0, 0, 14, 25, 7, 16, 7, 53, 59, 61, 17, 15]
for j in range(2, 12):
    n = n_text[j]
    for i in range(1, n + 1):
        russian_file = f"./datasets/book{j}/{j}_{i}_ru.txt"   # Глава на русском
        english_file = f"./datasets/book{j}/{j}_{i}_en.txt"   # Глава на английском

# Читаем файлы и получаем одну длинную строку для каждой главы
        russian_chapter = read_chapter(russian_file)
        english_chapter = read_chapter(english_file)

# Формируем запись для датасета. Можно добавить ключи "source" и "target" или "ru" и "en"
        example = {
            "source": english_chapter,
            "target": russian_chapter
        }

# Если нужно собрать датасет из нескольких пар, можно создавать список таких записей.
# Здесь для примера сохраняем один элемент в списке.
        dataset.append(example)

# Сохраняпшем датасет в JSON файл
with open("parallel_dataset.json", "w", encoding="utf-8") as out_file:
    json.dump(dataset, out_file, ensure_ascii=False, indent=2)

print("Датасет успешно сохранён в файл 'parallel_dataset.json'")