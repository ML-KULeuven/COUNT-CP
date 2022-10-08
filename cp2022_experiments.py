import json
import glob
import csv
import learner
import pickle
import logging
import time
from multiprocessing import Pool
import argparse
import itertools as it
from instance import Instance
from learn import learn, create_model
from instances.type06 import model_type06
from instances.nurse_rostering import nurse_rostering_instance
from experiments import true_model

logger = logging.getLogger(__name__)

def experiment_time_taken(instances, training_size, symbolic=True):
    ptype = instances[0].problem_type
    with open(f"type_{ptype:02d}_training_size_{training_size}_symbolic_{symbolic}_time.csv", "w") as csv_file:
        filewriter = csv.writer(csv_file, delimiter=",")
        filewriter.writerow(
            [
                "type",
                "training_size",
                "learning_time",
            ]
        )
        start = time.time()
        bounding_expressions = learn(instances[:1], training_size, symbolic)
        learning_time = time.time() - start
        print(learning_time)
        filewriter.writerow(
            [
                ptype,
                training_size,
                learning_time,
            ]
        )

def experiments(instances, training_size, symbolic=True):
    ptype = instances[0].problem_type
    with open(f"type_{ptype:02d}_training_size_{training_size}_symbolic_{symbolic}.csv", "w") as csv_file:
        filewriter = csv.writer(csv_file, delimiter=",")
        filewriter.writerow(
            [
                "type",
                "instance",
                "training_size",
                "total_constraints",
                "learned_constraints",
                "learning_time",
                "testing_time",
                "precision",
                "recall",
            ]
        )
        start = time.time()
        bounding_expressions = learn(instances[:-1], training_size, symbolic)
        learning_time = time.time() - start
        pickleVar = bounding_expressions

        for instance in instances:
            print(f"instance {instance.number}")
            learned_model, total_constraints = create_model(bounding_expressions, instance, propositional=False)
            print(f"number of constraints: {len(learned_model.constraints)}")
            start_test = time.time()
            precision, recall = learner.compare_models(learned_model, true_model(ptype, instance), instance)
            print(f"precision: {int(precision)}%  |  recall:  {int(recall)}%")
            filewriter.writerow(
                [
                    ptype,
                    instance.number,
                    training_size,
                    total_constraints,
                    len(learned_model.constraints),
                    learning_time,
                    time.time() - start_test,
                    precision,
                    recall,
                ]
            )
    pickle.dump(pickleVar, open(f"type_{ptype:02d}_training_size_{training_size}_symbolic_{symbolic}.pickle", "wb"))


if __name__ == "__main__":
    # types = [l for l in range(11, 17) if l != 9]
    # types = [int(sys.argv[1])]
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp", type=str, required=True)
    parser.add_argument("--training_size", type=int, nargs='*', default=[1, 5, 10])

    args = parser.parse_args()

    if args.exp == "nurses":
        train_instance1 = nurse_rostering_instance(10, 7, 5, 8)
        train_instance2 = nurse_rostering_instance(20, 7, 12, 15)
        test_instance1 = nurse_rostering_instance(25, 7, 15, 18)
        test_instance2 = nurse_rostering_instance(40, 7, 25, 28)
        instances = [train_instance1, train_instance2, test_instance1, test_instance2]
        symbolic = [True, False]
        iterations = list(
            it.product(
                [instances[:3]],
                args.training_size,
                # symbolic,
            )
        )
    else:
        if args.exp == "graph":
            ptype = 1
        elif args.exp == "sudoku":
            ptype = 6
        elif args.exp == "queens":
            ptype = 20
        elif args.exp == "magic":
            ptype = 21
        path = f"instances/type{ptype:02d}/inst*.json"
        files = sorted(glob.glob(path))
        instances = []
        for file in files:
            with open(file) as f:
                instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), ptype))
        if args.exp == "magic" or args.exp == "graph":
            instances = [instances[i] for i in [0, 3, 5]]
        if args.exp == "queens":
            instances = [instances[i] for i in [4, 5, 6]]
        iterations = list(
            it.product(
                [instances[:3]],
                args.training_size,
            )
        )
    for instance in instances:
        print(len(instance.pos_data))

    pool = Pool(processes=len(iterations))
    pool.starmap(experiment_time_taken, iterations)
