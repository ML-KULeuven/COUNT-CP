Repo of the CP22 paper:

  Mohit Kumar, Samuel Kolb, Tias Guns: Learning Constraint Programming Models from Data Using Generate-And-Aggregate. CP 2022: 29:1-29:16

PDF available at: https://drops.dagstuhl.de/opus/volltexte/2022/16658/pdf/LIPIcs-CP-2022-29.pdf


For questions please contact Prof. Tias Guns

## Paper history
We first created prototype code for the PTHG 21 Constraint Acquisition Challenge: https://freuder.wordpress.com/progress-towards-the-holy-grail-workshops/pthg-21-the-fifth-workshop-on-progress-towards-the-holy-grail/ 

We then refactored and extended the parts that we felt would make for an interesting paper. That is provided in this repo.

## Requirements
The core part of CountCP (`learn.py`) uses the constraint solving library CPMpy. At submission time CPMpy was at v0.9.7, but it could be that we were using an older version.

I hope everything continues to work with the latest CPMpy version, if not let us know so we can fix it.

We also use the SymPy library for symbolic expressions.

## Repo structure

* `instances/`  contains the 17 original instances from the PTHG21 challenge including our post-hoc interpretation of the ground truth model expressed in CPMpy (based on the names of the problems shown in the PTHG21 slides, copied into instances/problems.txt. We do not know the real ground-truth models), plus our own simple nurse rostering instance generator that outputs PTHG21 formatted data.
* `README.md` this file, send us updates if you feel something is missing
* `cp2022_experiments.py` tools for running and measuring the experiments in the CP22 paper. Check out the main part to see what arguments are sensible (e.g. graph, sudoku, queens, magic and nurses)
* `instance.py`  class(es) to help manage the PTHG21 instances
* `learn.py`  **This is the core of the code** and contains the learnin logic
* `musx.py`  CPMpy MUSX implementation from its advanced example directory
* `results.py`  helper code to create visualisations from the experiment logs

