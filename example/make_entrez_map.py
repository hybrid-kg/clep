# -*- coding: utf-8 -*-

"""Make Entrez ID mappings using Bio2BEL."""

import json

import bio2bel_hgnc


def main():
    """Make mapping files."""
    manager = bio2bel_hgnc.Manager()

    if not manager.is_populated():
        manager.populate()

    entrez_id_to_hgnc_id = manager.build_entrez_id_to_hgnc_id_mapping()
    with open('entrez_id_to_hgnc_id.json', 'w') as file:
        json.dump(entrez_id_to_hgnc_id, file, indent=2, sort_keys=True)

    entrez_id_to_hgnc_symbol = manager.build_entrez_id_to_hgnc_symbol_mapping()
    with open('entrez_id_to_hgnc_symbol.json', 'w') as file:
        json.dump(entrez_id_to_hgnc_symbol, file, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
