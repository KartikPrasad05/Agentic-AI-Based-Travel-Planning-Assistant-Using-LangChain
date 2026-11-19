# Travel Planning Assistant — Prepared Datasets

## Folder contents
- `data/raw/`: untouched copies of the three uploaded source CSVs.
- `data/processed/flights_clean.csv`: normalized flight records.
- `data/processed/hotels_clean.csv`: normalized hotel catalogue.
- `data/processed/places_clean.csv`: normalized tourist attraction records.
- `validation_report.json`: row counts, columns, missing-value summary and duplicate checks.
- `prepare_datasets.py`: reproducible cleaning script.

## Important data limitations
1. **Flights:** Historical fare dataset, not live flight inventory. `days_left` is a booking-window feature, not a calendar travel date. Prices must be labelled historical/reference prices.
2. **Hotels:** Goibibo records were crawled in 2016. No reliable nightly price field is present. Star rating `0` is converted to null in `star_rating` because it means unknown/unrated, not a genuine zero-star hotel. Original value is retained in `raw_star_rating`.
3. **Places:** The source contains 323 attraction rows. Derived columns such as `value_score` and `performance_score` are excluded from the cleaned file because their calculation/meaning is not documented in the supplied CSV. The original source remains preserved in `data/raw/`.
4. **No live booking claim:** These files support discovery, filtering and demonstrations only. Live availability and current prices require a suitable live API/provider.
5. **Licensing:** Verify each original dataset's licence and attribution terms before publishing source data or processed derivatives in a public repository.

## Integration policy
Keep the original project JSON files (`flights.json`, `hotels.json`, `places.json`) as separate baseline inputs. These CSVs are supplementary datasets. Do not overwrite the originals. For hotels, use the original JSON's nightly price for the project's budget calculation until a licensed, current price source is added.

## Rebuild
Run `python prepare_datasets.py` from this folder after installing pandas.
