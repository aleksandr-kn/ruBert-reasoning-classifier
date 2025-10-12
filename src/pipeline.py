import csv
import os
from datetime import datetime
import subprocess
import sys
import pandas as pd
import argparse
import random
import string

def create_run_directory(base_dir: str) -> str:
    """
    Создает уникальную директорию для текущего запуска пайплайна.
    Формат: run_<unique_id>_<YYYY-MM-DD_HH-MM-SS>

    Args:
        base_dir (str): Базовая директория для всех запусков.

    Returns:
        str: Путь к созданной директории для текущего запуска.
    """
    unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f"run_{unique_id}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[+] Run directory created: {run_dir} (unique_id: {unique_id})")
    return run_dir

def extract_texts_from_csv(input_csv_path: str, output_txt_path: str, text_column: str = "text") -> None:
    """
    Извлекает тексты из CSV-файла и сохраняет их в отдельный текстовый файл.

    Args:
        input_csv_path (str): Путь к исходному CSV-файлу.
        output_txt_path (str): Путь к выходному .txt файлу, куда будут записаны тексты.
        text_column (str, optional): Название колонки с текстами в CSV.
                                     По умолчанию 'text'.

    Raises:
        FileNotFoundError: Если CSV-файл не найден по указанному пути.
        KeyError: Если указанная колонка text_column отсутствует в CSV.
    """

    try:
        with open(input_csv_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
                for row in reader:
                    if text_column not in row:
                        raise KeyError(f"Колонка '{text_column}' не найдена в CSV")
                    text = row[text_column].strip()
                    txt_file.write(text + '\n')
        print(f"Тексты успешно извлечены в {output_txt_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV-файл не найден: {input_csv_path}")

def run_postagger(input_txt_path: str, output_txt_path: str, postagger_script_path: str = None) -> None:
    """
    Запускает postagging текстов через rusvectores Python скрипт,
    читая из input_txt_path и записывая результат в output_txt_path.
    """
    if postagger_script_path is None:
        postagger_script_path = os.path.join(
            os.path.dirname(__file__),
            "preprocessing/postagging/udpipe/rus_preprocessing_udpipe.py"
        )

    if not os.path.exists(postagger_script_path):
        raise FileNotFoundError(f"Postagger script not found: {postagger_script_path}")

    # Читаем input_txt и передаем в stdin скрипту
    with open(input_txt_path, 'r', encoding='utf-8') as f_in, \
         open(output_txt_path, 'w', encoding='utf-8') as f_out:

        subprocess.run(
            [sys.executable, postagger_script_path],
            stdin=f_in,
            stdout=f_out,
            stderr=sys.stderr,
            check=True,
            encoding='utf-8'
        )

    print(f"Postagged тексты сохранены в {output_txt_path}")

def merge_postagged_to_csv(original_csv_path: str, postagged_txt_path: str, output_csv_path: str,
                           text_column: str = "text") -> None:
    """
    Возвращает postagged тексты обратно в CSV.

    Args:
        original_csv_path (str): Путь к исходному CSV.
        postagged_txt_path (str): Путь к файлу с postagged текстами.
        output_csv_path (str): Куда сохранить новый CSV.
        text_column (str): Колонка, которую заменяем postagged текстами.
    """
    # Загружаем оригинальный CSV
    df = pd.read_csv(original_csv_path)

    # Загружаем постаггед тексты
    with open(postagged_txt_path, 'r', encoding='utf-8') as f:
        postagged_texts = [line.strip() for line in f]

    if len(df) != len(postagged_texts):
        raise ValueError("Количество строк в CSV и postagged тексте не совпадает!")

    # Заменяем колонку текстов на postagged
    df[text_column] = postagged_texts

    # Сохраняем новый CSV
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Postagged CSV сохранен в {output_csv_path}")

def build_word2index_npz(postagged_csv_path: str, output_npz_path: str, csv_to_word2index_script_path: str) -> None:
    """
    Строит word2index.npz из postagged CSV, используя готовый скрипт csv_to_word2index_npz.py.

    Args:
        postagged_csv_path (str): Путь к CSV с postagged текстами.
        output_npz_path (str): Куда сохранить word2index.npz.
        csv_to_word2index_script_path (str): Путь к скрипту csv_to_word2index_npz.py
    """
    if not os.path.exists(csv_to_word2index_script_path):
        raise FileNotFoundError(f"Script not found: {csv_to_word2index_script_path}")

    subprocess.run([
        sys.executable,
        csv_to_word2index_script_path,
        "--input", postagged_csv_path,
        "--output", output_npz_path,
        "--column", "text"
    ], check=True)

    print(f"[+] word2index сохранён в {output_npz_path}")

def build_embedding_matrix_npz(word2index_npz_path: str, rusvectores_model_path: str, output_npz_path: str) -> None:
    """
    Строит матрицу эмбеддингов W из словаря word2index и модели RusVectores.
    Сохраняет результат в .npz файл.

    Args:
        word2index_npz_path (str): Путь к .npz файлу с word2index.
        rusvectores_model_path (str): Путь к бинарной модели RusVectores (.bin).
        output_npz_path (str): Путь к выходному .npz файлу для embedding matrix.
    """
    # Абсолютный путь до скрипта
    script_path = os.path.join(
        os.path.dirname(__file__),
        "embeddings/build_embedding_matrix.py"
    )

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Вызываем скрипт через subprocess
    subprocess.run([
        sys.executable,
        script_path,
        "--input", word2index_npz_path,
        "--model", rusvectores_model_path,
        "--output", output_npz_path
    ], check=True)

    print(f"[✓] Матрица эмбеддингов построена и сохранена в {output_npz_path}")

def run_prepocessing_pipeline(
    source_csv_dataset_path: str = "data/texts/data_2025_20_07/file_no_duplicates_correct.csv",
    text_column: str = 'text'
) -> None:
    """
    Запускает pipeline для подготовки текста.
    По итогу должна сформировать 2 файла:
    - embedding_matrix.npz - матрица эмбеддингов
    - word2vec.npz - словарь - слово - индекс

    Параметры:
    - source_csv_dataset_path: путь до исходного CSV файла с текстами
    - text_column: имя колонки с текстом в CSV
    """

    # Базовая директория, куда складываются промежуточные файлы
    base_output_dir = "./outputs"

    # Создаем базовую директорию, куда будут склаывадться все промежуточные выводы:
    current_run_dir = create_run_directory(base_output_dir)

    extracted_txt_file_path = os.path.join(current_run_dir, "extracted_texts.txt")
    # Извлекаем тексты
    extract_texts_from_csv(source_csv_dataset_path, extracted_txt_file_path, 'text')

    # Шаг 2: Postagging для исходных текстов
    # Путь для postagged текстов
    postagged_txt_file_path = os.path.join(current_run_dir, "texts_postagged.txt")
    run_postagger(extracted_txt_file_path, postagged_txt_file_path)

    # Шаг 3: возвращаем обратно тексты в csv file, собираем postagged.csv
    postagged_csv_path = os.path.join(current_run_dir, "postagged_dataset.csv")
    merge_postagged_to_csv(
        original_csv_path=source_csv_dataset_path,
        postagged_txt_path=postagged_txt_file_path,
        output_csv_path=postagged_csv_path,
        text_column=text_column
    )

    # Шаг 4 - строим word2index.npz
    csv_to_word2index_script_path = os.path.join("src", "embeddings", "csv_to_word2index.py")
    word2index_npz_path = os.path.join(current_run_dir, "word2index.npz")

    build_word2index_npz(postagged_csv_path, word2index_npz_path, csv_to_word2index_script_path)

    # Шаг 5 - строим Embedding_matrix.npz
    embedding_matrix_npz_path = os.path.join(current_run_dir, "embedding_matrix.npz")

    # Путь до модели rusvectores
    rusvectores_model_path = './pretrained/ruscorpora_upos_cbow_300_20_2019/model.bin'

    build_embedding_matrix_npz(word2index_npz_path, rusvectores_model_path, embedding_matrix_npz_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text preprocessing pipeline")

    # Аргумент для пути до исходного CSV файла
    parser.add_argument(
        "--csv",                       # имя аргумента
        type=str,                       # ожидаемый тип
        default="data/texts/data_2025_20_07/file_no_duplicates_correct.csv",  # значение по умолчанию
        help="Path to the source CSV file"  # описание для --help
    )

    # Аргумент для имени колонки с текстом
    parser.add_argument(
        "--column",
        type=str,
        default="text",
        help="Name of the column containing text"
    )

    # Разбираем аргументы командной строки
    args = parser.parse_args()

    # Передаем полученные аргументы в функцию pipeline
    run_prepocessing_pipeline(
        source_csv_dataset_path=args.csv,  # путь до CSV
        text_column=args.column            # колонка с текстом
    )
