import pandas as pd
from clepp.embedding.network_generator import do_graph_gen


def kg_test():
    df = pd.read_csv(
        "/Users/bio/Work/pathway_modelling/clepp/src/clepp/sample_scoring/temp.csv",
        index_col=None,
        sep='\t'
    )
    kg = pd.read_csv(
        "/Users/bio/Work/pathway_modelling/clepp-resources/data/kg/hippie.tsv",
        header=0,
        sep='\t'
    )
    do_graph_gen(data=df, network_gen_method='Interaction Network', kg_data=kg)


# def gmt_test():
#     df = pd.read_csv("data4kg.csv", index_col=None, header=0)
#     kg = pd.read_csv("kg.csv", index_col=None, header=0)
#     do_graph_gen(data=df, embedding_method=1, kg_data=kg)
#
#
# def folder_test():
#     df = pd.read_csv("data4kg.csv", index_col=None, header=0)
#     kg = pd.read_csv("kg.csv", index_col=None, header=0)
#     do_graph_gen(data=df, embedding_method=1, kg_data=kg)
if __name__ == '__main__':
    kg_test()
