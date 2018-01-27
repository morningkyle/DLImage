import os

from classify import classify, load_graph, initialize_args


def get_sub_dirs(path):
    if os.path.isdir(path):
        return [d[0] for d in os.walk(path)]
    elif os.path.isfile(path):
        return [path]
    else:
        return []


def classify_dir_recursive():
    args = initialize_args()
    graph = load_graph(args.graph)
    directories = get_sub_dirs(args.image)
    for path in directories:
        args.image = path
        classify(graph, args)
    print("Classification Done!")


if __name__ == "__main__":
    classify_dir_recursive()



