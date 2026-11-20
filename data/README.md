# Источники данных

`scripts/fetch_data.py` загружает в `data/raw/` полные языковые части MGSM, Belebele и HumanEval. Версии, исходные URL, размеры и SHA-256 записаны в [sources.lock.json](sources.lock.json). Исходные TSV и JSONL дополнительно сжимаются в gzip. Размер и хеш проверяются по исходным байтам. Хеш сжатого контейнера приведён для справки, поскольку заголовок gzip может отличаться между версиями Python. Содержимое кеша исключено из Git.

| Источник | Авторство и материал | Лицензия |
| --- | --- | --- |
| MGSM | Shi et al., Google Research, *Language Models are Multilingual Chain-of-Thought Reasoners*. Английский и русский TSV | [CC BY 4.0](licenses/MGSM_CC_BY_4.0.txt) |
| Belebele | Bandarkar et al., Meta, *The Belebele Benchmark*. Английский и русский JSONL | [CC BY-SA 4.0](licenses/Belebele_CC_BY_SA_4.0.txt) |
| HumanEval | Chen et al., OpenAI, *Evaluating Large Language Models Trained on Code*. Английский JSONL.gz | [MIT](licenses/HumanEval_MIT.txt) |
| MGSM-Rev2 | Peter et al., Google, *Mind the Gap… or Not?*. Список замен для аудита исходных условий | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

Belebele используется для внешнего тестирования по [назначению авторов](https://github.com/facebookresearch/belebele#training-set). Производные метаданные Belebele и пометки из списка MGSM-Rev2 распространяются с атрибуцией и на условиях CC BY-SA 4.0. Условия использования остальных данных приведены в таблице. В HumanEval эталонная функция и скрытые тесты доступны для проверки, отдельно от условия, которое видит модель.

В [метаданных заданий](../reports/tables/task_metadata.csv) сохраняются идентификаторы, группы, длины и флаги качества. Полных условий и эталонов в таблице нет. `tokens_proxy` измерен cl100k_base. `duplicate_options` отмечает повторяющиеся варианты Belebele, `duplicate_correct_option` — повторы правильного варианта. `upstream_revised` отмечает строки исходного MGSM, изменённые в MGSM-Rev2. `eligible` отражает проверку полноты условия и эталона, формата эталона и повторов вариантов. Метки разбиения иллюстрируют влияние группировки.

Статьи [MGSM](https://arxiv.org/abs/2210.03057), [Belebele](https://aclanthology.org/2024.acl-long.44/), [HumanEval](https://arxiv.org/abs/2107.03374), [MGSM-Rev2](https://arxiv.org/abs/2511.05162).
