from pathlib import Path

import pandas as pd


def main() -> None:
    DIRPATH_INPUT = '../results/2020-06-24/'
    DIRPATH_OUTPUT = DIRPATH_INPUT
    threshold = None
    
    p = Path(DIRPATH_INPUT)
    list_filepath_input = list(p.glob('*.csv'))

    df_outlier = pd.DataFrame(columns=['objectid', 'mjd_st', 'mjd_en'])
    for filepath_input in list_filepath_input:
        df_outlier_alg = pd.read_csv(filepath_input)
        df_outlier = pd.concat([df_outlier, df_outlier_alg])

    df_ensemble = \
        df_outlier.groupby(df_outlier.columns.tolist()).size().reset_index()
    df_ensemble.rename(columns={0: 'records'}, inplace=True)
    
    if threshold is None:
        threshold = df_ensemble['records'].max()
    
    df_output = \
        df_ensemble[df_ensemble['records'] >= threshold].copy()
    df_output.drop('records', axis=1, inplace=True)
    df_output.sort_values(['objectid', 'mjd_st', 'mjd_en'], inplace=True)
    df_output.to_csv(f'{DIRPATH_OUTPUT}ensemble_{threshold}-or-more.csv',
                     index=False)


if __name__ == '__main__':
    main()
