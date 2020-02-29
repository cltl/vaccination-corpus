from __future__ import annotations

import pandas as pd
import csv
import os
from tqdm import tqdm
import click
from pathlib import Path
import pickle
import gzip
from loguru import logger

# Classes for CoNLL data
from conll_data import Token, Document
from typing import List

# configure log file
logger.add("read_annotations.log", rotation="100 MB")


COLUMNS = [
    "sent_id",
    "token_id",
    "word",
    "lemma",
    "pos",
    "head",
    "deprel",
    "offset_start",
    "offset_end",
    "event",
    "attr_content",
    "attr_source",
    "attr_cue",
    "claim",
]


def read_conll_file(conll_file: Path):
    """
    Returns a Document object representing the annotations in a given CoNLL file
    """
    df = pd.read_csv(
        conll_file,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        escapechar="\\",
        names=COLUMNS,
    )
    df = df.replace(to_replace={"_": None, "": None})
    lines_conll_file = df.to_dict("records")
    tokens = get_tokens(lines_conll_file)
    doc = Document(id=conll_file.stem, tokens=tokens)
    return doc


def get_tokens(lines_conll_file) -> List[Token]:
    """
    Returns a list of Token objects representing the annotations
    of each line in a CoNLL file
    """
    apred_keys = [key for key in lines_conll_file[0].keys() if key.startswith("apred")]

    tokens = []
    for token in lines_conll_file:
        if token["word"] is not None:

            apred_values = {key: token[key] for key in apred_keys if token[key]}

            token_dict = {
                key: value for key, value in token.items() if key not in apred_keys
            }
            token_dict["apred"] = apred_values
            t = Token(**token_dict)
            tokens.append(t)

    return tokens


@click.command()
@click.option(
    "--input",
    "input_dir",
    type=click.Path(exists=True),
    default="data/annotations-conll-claims-events-attribution",
    help="Directory containing the CoNLL annotations",
    show_default=True,
)
@click.option(
    "--outdir",
    "outdir",
    type=click.Path(file_okay=False, writable=True),
    default="data/annotations-pickle",
    help="Directory that will contain the Document objects as pickle files",
    show_default=True,
)
def main(input_dir: str, outdir: str):

    # if necessary, create output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # read conll files and create Document instances
    logger.info("Reading annotations from CoNLL files")
    conll_files = [x for x in Path(input_dir).glob("*.conll.annot")]
    for conll_file in tqdm(conll_files):
        doc = read_conll_file(conll_file)

        # write to output directory
        outfile_doc = os.path.join(outdir, doc.id) + ".pickle.gz"
        with gzip.open(outfile_doc, "wb") as outfile:
            pickle.dump(doc, outfile)
    logger.info("Done!")


if __name__ == "__main__":
    main()
