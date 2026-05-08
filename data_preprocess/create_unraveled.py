import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

UNRAVELED_FLOWS_PATH = os.path.join('..', 'datasets', 'cherry-picked', 'unraveled', 'network-flows')

def load_and_concatenate_csvs(path):
    unraveled_flows = pd.DataFrame()
    failed_files = []

    for directory in os.listdir(path):
        for filename in os.listdir(os.path.join(path, directory)):
            if filename.endswith(".csv"):
                file_path = os.path.join(path, directory, filename)
                try:
                    df = pd.read_csv(file_path, on_bad_lines='skip')
                    unraveled_flows = pd.concat([unraveled_flows, df], ignore_index=True)
                    # print(f"Carregado: {directory}/{filename}")
                except Exception as e:
                    print(f"Erro ao carregar: {directory}/{filename}")
                    print(f"Detalhes: {str(e)}")
                    failed_files.append((file_path, str(e)))

    print(f"Arquivos com erro: {len(failed_files)}")

    if failed_files:
        print("Arquivos com problemas:\n")
        for path, error in failed_files:
            print(f"  - {path}: {error}")

    print(unraveled_flows.head())
    print(unraveled_flows.info())

    return unraveled_flows

def remove_outliers_zscore(df):
    before_remove = len(df)

    outlier_mask = pd.Series(False, index=df.index)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        z_scores = np.abs(stats.zscore(df[col]))
        outlier_mask |= (z_scores >= 4)

    df_without_outliers = df[~outlier_mask]
    after_remove = len(df_without_outliers)

    print(f'{before_remove - after_remove} outliers removed')

    return df_without_outliers

def remove_outliers_isolation_forest(df, contamination=0.01, random_state=42):
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)

    df_clean = df.copy()
    numeric_features = {}
    for column in df_clean.columns:
        converted = pd.to_numeric(df_clean[column], errors='coerce')
        if converted.notna().any():
            numeric_features[column] = converted

    if not numeric_features:
        return df_clean

    features = pd.DataFrame(numeric_features)
    medians = features.median(numeric_only=True)
    features = features.fillna(medians)

    if features.empty or len(features) < 2:
        return df_clean

    predictions = iso_forest.fit_predict(features.astype(float))
    return df_clean.loc[predictions == 1].copy()

def preprocess_dataset(df):
    preprocessed = df.copy()

    if 'Stage' in preprocessed.columns:
        stage_series = preprocessed['Stage'].astype(str).str.strip()
        normal_mask = stage_series.str.lower().eq('normal')
        preprocessed['Stage'] = stage_series
        preprocessed.loc[normal_mask, 'Stage'] = 'Benign'

    return preprocessed

def group_and_sort(df):
    required_columns = ['Stage', 'src_ip', 'bidirectional_first_seen_ms']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    grouped_sorted = df.copy()
    grouped_sorted['bidirectional_first_seen_ms'] = pd.to_numeric(
        grouped_sorted['bidirectional_first_seen_ms'], errors='coerce'
    )
    grouped_sorted = grouped_sorted.dropna(subset=['Stage', 'src_ip', 'bidirectional_first_seen_ms'])
    grouped_sorted = grouped_sorted.sort_values(
        by=['Stage', 'src_ip', 'bidirectional_first_seen_ms'], kind='stable'
    ).reset_index(drop=True)
    return grouped_sorted

def create_delta_t(df):
    transformed = df.copy()
    first_seen_columns = [col for col in transformed.columns if 'first_seen_ms' in col]

    if not first_seen_columns:
        return transformed

    for column in first_seen_columns:
        transformed[column] = pd.to_numeric(transformed[column], errors='coerce')
        prefix = column.split('first_seen_ms')[0].rstrip('_')
        delta_column = f"delta_t_{prefix}" if prefix else 'delta_t'
        
        transformed[delta_column] = transformed[column].diff().fillna(0)

    drop_columns = [
        col for col in transformed.columns
        if ('first_seen' in col.lower()) or ('last_seen' in col.lower())
    ]
    transformed = transformed.drop(columns=drop_columns, errors='ignore')
    return transformed

def save_dataframe(df, output_path, output_format):
    if output_format == 'csv':
        df.to_csv(output_path, index=False)
    elif output_format == 'parquet':
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def save_by_stage(df, output_dir, output_format):
    os.makedirs(output_dir, exist_ok=True)
    for stage, stage_df in df.groupby('Stage', sort=False):
        stage_name = str(stage).strip().replace(' ', '_').replace('/', '_')
        extension = 'csv' if output_format == 'csv' else 'parquet'
        stage_output_path = os.path.join(output_dir, f"stage_{stage_name}.{extension}")
        save_dataframe(stage_df, stage_output_path, output_format)

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default=UNRAVELED_FLOWS_PATH)
    parser.add_argument('--outlier-method', choices=['none', 'zscore', 'isolation-forest'], default='isolation-forest')
    parser.add_argument('--contamination', type=float, default=0.01)
    parser.add_argument('--save-mode', choices=['single', 'by-stage'], default='single')
    parser.add_argument('--output-format', choices=['csv', 'parquet'], default='csv')
    parser.add_argument('--output-path', default=os.path.join('..', 'datasets', 'cherry-picked', 'unraveled', 'processed_unraveled.csv'))
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    df = load_and_concatenate_csvs(args.input_path)
    df = preprocess_dataset(df)

    if args.outlier_method == 'zscore':
        df = remove_outliers_zscore(df)
    elif args.outlier_method == 'isolation-forest':
        df = remove_outliers_isolation_forest(
            df,
            contamination=args.contamination,
        )

    df = group_and_sort(df)
    df = create_delta_t(df)

    if args.save_mode == 'single':
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_dataframe(df, args.output_path, args.output_format)
    else:
        save_by_stage(df, args.output_path, args.output_format)

if __name__ == '__main__':
    main()



