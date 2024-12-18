# Решение для хакатона Alpha Hack от Альфа-Банка

## О проекте

Данный проект разработан в рамках хакатона **Alpha Hack**, организованного Альфа-Банком.  
Проект состоит из двух частей:
1. **Отборочный этап**: решение задачи табличного машинного обучения. Код представлен в файле `alpha-hack-fit-predict.py`.
2. **Финал**: решение задачи автоматизированного машинного обучения (AutoML) для 9 различных датасетов. Результаты команд оценивались по скору на этих наборах данных. Финальное решение представлено в `main.py`.

### Решение

В рамках решения были реализованы следующие этапы:
- **Понижение размерности**: использование метода PCA для сокращения числа признаков.
- **Генерация новых фичей**: арифметические операции для создания новых информативных признаков.
- **Отбор признаков**: применение Boruta для выбора наиболее значимых переменных.
- **Стекинг моделей**: ансамбль бустингов с метамоделью линейной регрессии.
- **Блендинг**: объединение решения выше с — **LightAutoML** и **AutoGluon**.

### Почему проект важен?

Проект демонстрирует:
- Гибридный подход к машинному обучению, совмещающий автоматизацию и ручной контроль над обработкой данных.
- Возможность эффективно работать с данными разных структур и объемов.
- Применимость в реальных бизнес-задачах, таких как кредитный скоринг, прогнозирование транзакций и анализ пользовательского поведения.

Проект выделяется своей адаптивностью к задачам различной сложности и конкурентоспособностью решений, проверенной на нескольких датасетах.

---

## Установка и запуск

Для воспроизведения решения выполните следующие шаги:

1. Установите все зависимости:
   ```bash
   pip install -r requirements.txt
Запустите соответствующие Jupyter-ноутбуки для задачи бинарной классификации.

## Технологии
- ![Python](https://img.shields.io/badge/Python-3.x-blue) — язык программирования для реализации всех компонентов.
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24-orange) — инструменты для обработки данных, построения моделей и их оценки.
- ![LightAutoML](https://img.shields.io/badge/LightAutoML-blueviolet) — библиотека для автоматического машинного обучения.
- ![AutoGluon](https://img.shields.io/badge/AutoGluon-darkgreen) — мощный инструмент для AutoML с поддержкой различных задач.


## Команда разработки
- Адиль Хабибуллин (БПМ 23-2) — ML
- Калинин Алексей (БИВТ-23-10) — ML
- Нестеров Никита (БИВТ-23-1) — ML
- Кайков Дмитрий (БИВТ-23-9) — ML

## Лицензия
Данный проект распространяется под лицензией MIT. Вы можете использовать, изменять и распространять проект в соответствии с условиями лицензии.
