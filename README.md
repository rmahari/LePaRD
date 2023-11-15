# LePaRD: A Large-Scale Dataset of Judges Citing Precedents

This repo contains the dataset introduced in the following paper:

[LePaRD: A Large-Scale Dataset of Judges Citing Precedents](tbd)

As of November 15, 2023 we are releasing a small version of the dataset and plan to release the full dataset, along with additional analyses, upon publication.

Download the dataset using [this link](https://drive.google.com/drive/folders/1TLdgeWBRQ2l1CpRYDmp8ppImrm87rHV3?usp=sharing)

# Reference

Please cite the following paper if you use LePaRD:

```bibtex
@article{mahari2023LePaRD,
  title={LePaRD: A Large-Scale Dataset of Judges Citing Precedents},
  author={Mahari, Robert and Stammbach, Dominik and Ash, Elliott and Pentland, Alex'Sandy'},
  journal={arXiv preprint},
  year={2023}
}
```


# Description

LePaRD is a massive collection of U.S. federal judicial citations to precedent in context. LePaRD builds on millions of expert decisions by extracting quotations to precedents from judicial opinions along with the preceding context. Each row of the dataset corresponds to a quotation to prior case law used in a certain context.

- passage_id: A unique idenifier for each passage
- destination_context: The preceding context before the quotation
- passage_text: The text of the passage that was quoted
- court: The court from which the passage originated
- date: The date when the opinion from which the passage originated was published

Contact [Robert Mahari](www.robertmahari.com) in case of any questions.


