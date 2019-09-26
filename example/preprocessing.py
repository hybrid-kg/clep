# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from xml.etree import ElementTree
import json
import click
import logging


design_raw_path = 'phenodata.txt'
miniml_xml_path = 'GSE65682_family.xml'
design_out_path = 'targets.txt'
namespace_string = 'http://www.ncbi.nlm.nih.gov/geo/info/MINiML'

platform_design_path = 'GPL13667-15572.txt'
exp_path = 'exp.txt'
exp_out_path = 'exp.txt'

logger = logging.getLogger(__name__)


@click.group()
def main():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@main.command()
def make_targets():
    design_raw = pd.read_csv(design_raw_path, sep='\t')
    xml = ElementTree.parse(miniml_xml_path)
    ns = {'namespace': namespace_string}
    root = xml.getroot()

    tags = ['pneumonia diagnoses', 'mortality_event_28days', 'time_to_event_28days']

    for i in tags:
        temp_lst = []
        for char in root.findall(f"./namespace:Sample/namespace:Channel/namespace:Characteristics[@tag='{i}']", ns):
            temp_var = char.text.replace('\n', '').strip()
            temp_lst.append(np.nan if temp_var == 'NA' else temp_var)
        design_raw[f'{i}'] = temp_lst

    design_matrix = design_raw[design_raw['pneumonia diagnoses'].isin(('cap', 'no-cap', 'hap'))].copy()

    survivor_mapping = {
        np.nan: np.nan,
        '0': 'SS',
        '1': 'SNS',
        }
    control_mapping = {
        'no-cap': 'Control',
        }

    design_matrix['Target'] = design_matrix['mortality_event_28days'].map(survivor_mapping)
    design_matrix['Control'] = design_matrix['pneumonia diagnoses'].map(control_mapping)

    final_design = pd.DataFrame(data=design_matrix.FileName, columns=['FileName'])

    final_design['Target'] = pd.concat([design_matrix['Target'].dropna(),
                                        design_matrix['Control'].dropna()]
                                       ).reindex_like(design_matrix).tolist()

    final_design = final_design[~final_design.Target.isna()]

    final_design.reset_index(inplace=True, drop=True)

    final_design.to_csv(design_out_path, sep='\t')


@main.command()
def make_exp():
    platform = pd.read_csv(platform_design_path, sep='\t', index_col=0, header=43)
    exp = pd.read_csv(exp_path, sep='\t')
    exp.rename(columns={'Unnamed: 0': 'genes'}, inplace=True)

    platform['Entrez Gene'] = [gene.split(' /// ')[0] for gene in platform['Entrez Gene']]
    probe2gene_mapping = dict(zip(platform.index, platform['Entrez Gene']))

    exp['genes'] = exp['genes'].map(probe2gene_mapping)

    exp = exp[exp.genes != '---']

    with open('../example/entrez_id_to_hgnc_symbol.json') as f:
        entrez_id_to_hgnc_symbol = json.load(f)

    exp['genes'] = exp['genes'].astype('str')
    exp['genes'] = exp['genes'].map(entrez_id_to_hgnc_symbol)

    exp = exp[exp.genes != np.nan]

    exp.to_csv(exp_out_path, sep='\t', index=False)


if __name__ == '__main__':
    main()
