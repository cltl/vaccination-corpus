import click
from pathlib import Path
import os
from loguru import logger


def mask_conll(input_file: Path, output_file: Path):
    """Masks the words and lemmas in a CoNLL file.
    """

    # read conll file
    with open(input_file, "r") as infile:
        content = infile.read()

    # create new lines with masked word and lemma
    new_lines = []
    for line in content.split("\n"):
        columns = line.split("\t")
        if len(columns) > 1:
            columns[2] = "_"  # word
            columns[3] = "_"  # lemma
            new_line = "\t".join(columns)
        else:
            new_line = line
        new_lines.append(new_line)
    new_content = "\n".join(new_lines)

    # write to output file
    with open(output_file, "w") as outfile:
        outfile.write(new_content)


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
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, writable=True),
    default="data/annotations-conll-mask2",
    help="Directory that will contain the masked CoNLL annotations",
    show_default=True,
)
def main(input_dir: Path, output_dir: Path):

    # if necessary, create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read conll files and create Document instances
    logger.info("Masking CoNLL files")
    conll_files = [x for x in Path(input_dir).glob("*.conll.annot")]

    for conll_file in conll_files:
        output_file = Path(output_dir).joinpath(conll_file.name)
        mask_conll(conll_file, output_file)

    logger.info("Done!")


if __name__ == "__main__":
    main()
