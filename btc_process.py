import pandas as pd

def filter_to_5min_intervals_keep_unix(input_file='BTC_RAW.csv', output_file='BTC_5min.csv'):
    df = pd.read_csv(input_file)
    
    # Convert the numeric UNIX timestamps to datetime temporarily for filtering
    df['Datetime_dt'] = pd.to_datetime(df['Datetime'], unit='s', errors='coerce')
    
    # Drop rows with invalid datetime conversion
    df = df.dropna(subset=['Datetime_dt'])
    
    # Filter for minutes divisible by 5 and seconds == 0
    mask = (df['Datetime_dt'].dt.minute % 5 == 0) & (df['Datetime_dt'].dt.second == 0)
    df_filtered = df[mask].copy()
    
    # Drop the temporary datetime column
    df_filtered = df_filtered.drop(columns=['Datetime_dt'])
    df_filtered = df_filtered.iloc[150_000:].reset_index(drop=True)
    # Save with original UNIX timestamp column intact
    df_filtered.to_csv(output_file, index=False)
    print(f"Saved filtered 5-minute interval data with UNIX timestamps to {output_file}")


filter_to_5min_intervals_keep_unix()
