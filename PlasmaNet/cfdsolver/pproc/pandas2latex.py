import argparse
import pandas as pd

columns_ordered = [
    'instability_de',
    'instability_de_value',
    'error_spectral_de',
    'instability_max',
    'instability_max_value',
    'error_spectral_max'
]

format_dict = {
    "instability_de": "{:.0f}".format,
    "instability_de_value": "{:.2f}".format,
    "instability_max": "{:.0f}".format,
    "instability_max_value": "{:.2f}".format,
    "error_spectral_de": "{:.2e}".format,
    "error_spectral_max": "{:.2e}".format
}

renaming_dict = {
    r'instability\_de\_value': '$(t/T_p)^\mrm{DA}$',
    r'instability\_de': r'\% DA unstable',
    r'instability\_max\_value': '$(t/T_p)^\mrm{max}$',
    r'instability\_max': r'\% max unstable',
    r'error\_spectral\_de': r'$\bar{\veps}$',
    r'error\_spectral\_max': r'$\veps^\mrm{max}$'
}


def frame_to_latex(filename, groupname, groupby_name):
    """ Convert pandas DataFrame into latex table """
    df = pd.read_hdf(filename, groupname)
    df_group = df.groupby(groupby_name).mean()
    latex_dataset = df_group.iloc[:, [2, 3, 7, 9, 10, 11]]
    latex_dataset = latex_dataset.reindex(columns=columns_ordered)
    latex_dataset.iloc[:, 0] *= 100
    latex_dataset.iloc[:, 3] *= 100
    latex_dataset.to_latex('tab_tmp.tex', formatters=format_dict, column_format='lcccccc',
                            label='Training datasets comparison', na_rep='-')
    
    fin = open('tab_tmp.tex', 'r')
    fout = open('tab.tex', 'w')

    lines = fin.readlines()
    fin.close()
    tmp_line = lines[5]
    for header_old, header_new in renaming_dict.items():
        tmp_line = tmp_line.replace(header_old, header_new)
    lines[5] = tmp_line
    fout.writelines(lines)


if __name__ == '__main__':
    args = argparse.ArgumentParser(
                description='Conversion of pandas DataFrame to Latex table')
    args.add_argument('-f', '--filename', required=True, type=str, help='Filename to write on')
    args.add_argument('-g', '--groupname', required=True, type=str, help='Groupname in HDF5 file')
    args.add_argument('-b', '--groupby_name', required=True, type=str, help='Filter for the DataFrame')
    args = args.parse_args()

    frame_to_latex(args.filename, args.groupname, args.groupby_name)
