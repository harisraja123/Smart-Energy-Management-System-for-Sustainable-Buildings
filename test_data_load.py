# test_data_load.py
from data_manager import DataManager

if __name__ == '__main__':
    dm = DataManager(timestamp_col='timestamp')
    df = dm.load_dataset('citylearn_dataset')

    # 1. Check index
    print("Index type:", type(df.index))
    print("Is monotonic:", df.index.is_monotonic_increasing)
    print("Index sample:", df.index[:5])

    # 2. Check overall shape and columns
    print("\nShape:", df.shape)
    print("First 10 columns:", df.columns.tolist()[:10])

    # 3. Ensure building, weather, carbon prefixes are present
    prefixes = {'building', 'weather', 'carbon_intensity'}
    present = {p for p in prefixes if any(c.startswith(p) for c in df.columns)}
    print("\nDetected prefixes in columns:", present)

    # 4. Spot-check derived features
    derived = [
        'hour', 'day_of_week', 'season',
        'building_1_temperature_mean_1h',
        'weather_outdoor_dry_bulb_temperature_mean_6h',
        'carbon_intensity_carbon_intensity_anomaly'
    ]
    print("\nDerived feature columns present:", [c for c in derived if c in df.columns])

    # 5. Peek at the head
    print("\nDataFrame head:")
    print(df.head())
