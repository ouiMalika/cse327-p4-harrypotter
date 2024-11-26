import csv
import time
import chainbased
import argparse
from helpers.prints import clear_line, print_progress_bar
import termwalk
import autoencoder
from basictypes import Atom, Predicate, Variable
from copy import copy
import nnunifier
import kbparser
import random
import atomgenerator
import numpy as np
# import takeMapping
from kbparser import parse_KB_file
from knowledgebase import KnowledgeBase, Rule, Path, generate_random_KB
import reasoner
import torch
import os

from vocab import Vocabulary
# need this to fix interrupt issues, must be before any scipy is imported
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

# program at large function: using a KB and set of examples, it performs backward chaining
# to infer new facts and rules based on the given inputs

global node_count

DEBUG = False
DEFAULT_KB_SIZE = 150
MAX_TRAIN_ANSWERS = 1000
# DEFAULT_KB_SIZE = 100

cache = reasoner.CachedUnify()


def clean_atom(atom: reasoner.Atom):
    for i in range(atom.arity):
        if isinstance(atom.arguments[i], reasoner.Variable):
            atom.arguments[i].name = atom.arguments[i].name[:2]
    return atom


# takes a rule and calls clean_atom on head and body
def clean_rule(rule: Rule):
    clean_atom(rule.head)
    for i in range(len(rule.body)):
        clean_atom(rule.body[i])

# Code is not longer used
# def gen_KB_text_files(KB: KnowledgeBase, KB_path, facts_path):
#     """
#     Takes a knowledge base and two file paths; generates text files
#     :param KB:
#     :param KB_path:
#     :param facts_path:
#     :return:
#     """
#     kbparser.KB_to_txt(KB, KB_path)
#     facts_list = reasoner.forwardchain(KB)
#     # TODO: clean_rule truncates variables to two characters. This seems like a bad design!!!
#     for fact in facts_list:
#         clean_rule(fact)
#     kbparser.KB_to_txt(KnowledgeBase(list(facts_list)), facts_path)


def gen_all_facts(KB: KnowledgeBase, vocab: Vocabulary) ->  (list[Atom], int):
    """
    Given a knowledge base, generates all possible facts and writes them to a file.

    :param KB: A knowledge base
    :param vocab: The vocabulary to use
    """
    facts, max_depth = reasoner.forwardchain(KB, vocab)
    facts_list = list(facts)
    # Note, if the KB has any rules where the head contains a variable not in the body,
    # then some facts may not be ground. However, we are currently sanitizing in forwardchain(),
    # so no need to sanitize here.
    # for i in range(len(facts_list)):
    #    facts_list[i] = vocab.sanitize_rule(facts_list[i])
    return facts_list, max_depth


def one_hot_encode_query(queries: dict[Atom, Rule], vocab: Vocabulary):
    """ Given a dictionary of queries, returns a list of tensors with their one-hot encodings.
    BUG? appears to only return the tensor for the last query currently.

    :param queries: dictionary with query keys and rule values
    :return: a list of three onehot vectors: goal atom, head, and body
    """
    for node in queries:
        rule_args = []
        # do we need to deepcopy here???
        # rule = deepcopy(queries[node])
        # node = deepcopy(node)
        # clean_rule(rule)
        # clean_atom(node)

        rule = vocab.sanitize_rule(queries[node])
        node = vocab.sanitize_atom(node)

        atom = vocab.oneHotEncoding(node)
        head = vocab.oneHotEncoding(rule.head)
        for arg in rule.body:
            rule_args.append(vocab.oneHotEncoding(arg))
        # TODO: Bug? doesn't this only return the encoding for the last query?
        encoding = [atom, head, rule_args]
    return encoding



# One-hot-encoding method to take predicate arity of atom,
# from translation, sort it, then translate by passing it into
# one-hot-encoding method.
# replaced by Vocab.oneHotEncoding
# def one_hot_encode_symbol_set(atoms: list[Atom]):
#     encodings = []
#     for atom in atoms:
#         enc = atomgenerator.onehotencoding(atom)
#         encodings.append(enc)
#     return encodings
# takes file paths and max depth. loads a KB, generates random query, and performs backward chaining on KB to get answers
# returns 2 lists of encodings of best and other (ideally worst) paths
# Ben: this function doesn't seem to be called anywhere in the project

# NOT USED
# def gen_KB_encodings(facts_path, KB_path, max_depth):
#     KB = parse_KB_file(KB_path)
#     facts_list = parse_KB_file(facts_path).rules
#     query = reasoner.gen_random_query(facts_list)
#     path = Path(query, None, 0)
#     answers = reasoner.backwardchain(KB, path, reasoner.MaxDepth(max_depth))
#     for z in answers:
#         for key in z:
#             # print(str(key))
#             # print(str(z[key]))
#             continue

#     path.calc_sf()
#     best, other = path.get_best_path()

#     return best, other


# loads a KB and a facts list. generates random query and performs backward chaining on KB to get answers
# returns a set of example encodings
def gen_example_encodings(seq, query, KB: KnowledgeBase, make_neg_facts: bool = True):
    # JDH: changed param from path to KB, so it doesn't have to reload the file for each query
    # global examples_list        # define a global variable named examples_list
    examples_list = []

    # KB = parse_KB_file(KB_path)
    print(str(seq+1) + ": " + str(query))

    restarts = 0
    ans_count = 0
    while ans_count == 0 and restarts < 3:
        if restarts > 0:
            clear_line()
            print("Restart...")
        path = Path(query, None, None, 0, [query])  # type: ignore
        examples = set()
        ans_count = backwardchain(KB, path, reasoner.MaxDepth(
            10), examples, examples_list, make_neg_facts)
        if ans_count == 0:
            restarts = restarts + 1
    clear_line()
    print("Answers: " + str(ans_count))
    print()
    return examples_list


def generate_unification_embeddings(examples_list: list,
                                    device: str,
                                    vocab: Vocabulary,
                                    embed_size: int,
                                    model_path: str = "rKB_model.pth") -> torch.Tensor:
    # Load the model once
    input_size = len(vocab.predicates) + \
        ((len(vocab.variables) + len(vocab.constants)) * vocab.maxArity)
    print("Embedding input size: " + str(input_size))
    model = nnunifier.NeuralNet(input_size,
                                nnunifier.hidden_size1,
                                nnunifier.hidden_size2,
                                embed_size).to(device)
    model.load_state_dict(torch.load(model_path,
                                     map_location=torch.device(device)))
    model.eval()

    # Initialize a list to collect embeddings
    embedding_list = []

    # Process examples in batches
    batch_size = 500
    i = 0
    for i in range(0, len(examples_list), batch_size):
        batch_examples = examples_list[i:i+batch_size]

        # Prepare embeddings for the entire batch
        batch_embeddings = [prep_model_example(
            example, vocab, model, device, embed_size).to(device) for example in batch_examples]

        # Collect embeddings in the list
        embedding_list.extend(batch_embeddings)

        # Print progress
        print_progress_bar(i, len(examples_list), length=15,
                           suffix='Prepping examples')

    if i < len(examples_list) - 1:
        remaining_examples = examples_list[i:]
        i += len(remaining_examples)
        remaining_embeddings = [prep_model_example(
            example, vocab, model, device, embed_size).to(device) for example in remaining_examples]
        embedding_list.extend(remaining_embeddings)
        print_progress_bar(i, len(examples_list), length=15,
                           suffix='Prepping examples')
    print()
    # Concatenate all embeddings at once
    embeddings = torch.cat(embedding_list, dim=0)
    embeddings = torch.unique(embeddings, dim=0)
    return embeddings


def generate_autoencoder_embeddings(examples_list: list,
                                    device: str,
                                    vocab: Vocabulary,
                                    model_path: str = "auto_encoder.pth") -> torch.Tensor:
    embeddings = torch.empty(0).to(device)
    whole_model = autoencoder.NeuralNet().to(device)
    whole_model.load_state_dict(torch.load(model_path,
                                           map_location=torch.device(device)))
    whole_model.eval()
    model = whole_model.encoder
    for example in examples_list:
        # Embed size set to 20 for now, but should be set to the actual size of the embedding
        emb = prep_model_example(example, vocab, model, device, 20).to(device)
        embeddings = torch.cat((embeddings, emb))
    embeddings = torch.unique(embeddings, dim=0)
    return embeddings


def generate_termwalk_embeddings(examples_list: list,
                                 device: str,
                                 vocab: Vocabulary) -> torch.Tensor:
    # Apparently, concatenation doesn't work with an empty sparse
    # tensor, even though it works for empty dense tensors
    embeddings = prep_termwalk_example(
        examples_list.pop(), vocab).to(device)
    for example in examples_list:
        emb = prep_termwalk_example(
            example, vocab).to(device)
        embeddings = torch.cat((embeddings, emb))
#    embeddings = torch.unique(embeddings.cpu().to_dense(), dim=0).to_sparse().to(device)
    return embeddings


def generate_chainbased_embeddings(examples_list: set | list,
                                   device: str="cpu", embed_size=20) -> torch.Tensor:
    embeddings = torch.empty(0).to(device)
#    time_list = []
    for example in examples_list:
        #        start = time.time()
        emb = prep_chainbased_example(example, embed_size)
#        end = time.time()
#        time_list.append(end - start)
        emb = emb.to(device)
        embeddings = torch.cat((embeddings, emb))
#    embeddings = torch.unique(embeddings, dim=0)
#    with open("chainbased_embedding_time.txt", "a") as file:
#        for i in time_list:
#            file.write(str(i) + " ")
#        file.write(str(sum(time_list) / len(time_list)) + "\n")
    return embeddings


def prep_examples(train_example_path: str,
                  KB_path: str = "randomKB.txt",
                  train_queries_path: str = "train_queries.txt",
                  make_neg_facts: bool = True):
    """ Given an input file of training queries, generates (goal/rule/score) examples for the training set
    for the guided reasoner. Saves results CSV files.

    :param train_example_path:
    :param KB_path:
    :param train_queries_path:
    :param make_neg_facts: Set to True to create negative training examples for goals that are resolved by facts
    :return:
    """
    print("running training queries...")
    if make_neg_facts:
        print("Negative facts will be generated")
    KB = kbparser.parse_KB_file(KB_path)

    examples_list = []
    async_results = []

    train_queries = [query.head for query in kbparser.parse_KB_file(
        train_queries_path).rules]

    # replace with this code when debugging
    for i in range(len(train_queries)):
        # cProfile.runctx(
        #     'gen_example_encodings(i, train_queries[i], KB)', globals(), locals(), sort='cumulative')
        async_results.append(gen_example_encodings(
            i, train_queries[i], KB, make_neg_facts))

    for i in range(len(async_results)):
        examples_list = examples_list + async_results[i]

    print("Before removing duplicates: " +
          str(len(examples_list)) + " examples")
    examples_list = set(examples_list)
    examples_list = list(examples_list)
    print("After removing duplicates: " +
          str(len(examples_list)) + " examples")

    pos_example_list = [e for e in examples_list if 1.0 == e[2]]
    neg_example_list = [e for e in examples_list if 0.0 == e[2]]

    # Duplicating positives if there are more negatives
    if len(neg_example_list) >= 2 * len(pos_example_list):
        dupe_times = (len(neg_example_list) // len(pos_example_list)) - 1
        print("Replicating pos examples " + str(dupe_times) + " times")
        new_pos = pos_example_list
        for i in range(dupe_times):
            new_pos = new_pos + pos_example_list
        pos_example_list = new_pos

    while len(pos_example_list) < len(neg_example_list):
        pos_dupe = random.choice(pos_example_list)
        pos_example_list = pos_example_list + [pos_dupe]

    # Duplicating negatives if there are more positives
    if len(pos_example_list) >= 2 * len(neg_example_list):
        dupe_times = (len(pos_example_list) // len(neg_example_list)) - 1
        print("Replicating neg examples " + str(dupe_times) + " times")
        new_neg = neg_example_list
        for i in range(dupe_times):
            new_neg = new_neg + neg_example_list
        neg_example_list = new_neg

    while len(neg_example_list) < len(pos_example_list):
        neg_dupe = random.choice(neg_example_list)
        neg_example_list = neg_example_list + [neg_dupe]

    print("pos len=" + str(len(pos_example_list)) +
          ", neg len=" + str(len(neg_example_list)))

    examples_list = [*pos_example_list, *neg_example_list]

    with open(train_example_path, mode='w', newline='') as exout_file:
        fieldnames = ['goal', 'rule', 'score']
        writer = csv.DictWriter(
            exout_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        if exout_file.tell() == 0:
            writer.writeheader()
        for example in examples_list:
            ex_row = {'goal': example[0],
                      'rule': example[1], 'score': example[2]}
            writer.writerow(ex_row)


def prep_model_example(example, vocab: Vocabulary, model, device, embed_size: int) -> torch.Tensor:
    """ Given a goal/rule training example generates embeddings using a specified embedding model

    :param example: list of goal, rule, and score
    :param model: the embedding model to use
    :param device: cpu or cuda
    :return: PyTorch tensor of the embedding
    """
    with torch.no_grad():
        # put in a dictionary with goal as key, rule as value
        query = {example[0]: example[1]}
        score = example[2]
        query = one_hot_encode_query(query, vocab)
        # TODO: is all of this conversion to cpu inefficient for cuda machines?
        atom = model(torch.FloatTensor(query[0]).to(
            device)).cpu().detach().numpy()
        rule_head = model(torch.FloatTensor(
            query[1]).to(device)).cpu().detach().numpy()
        args = torch.zeros(embed_size).to(device)
        rule_args = query[2]
        for arg in rule_args:
            arg = model(torch.FloatTensor(arg).to(device))
            args = torch.add(args, arg)
        args = args.cpu().detach().numpy()
        embedding = torch.from_numpy(
            np.array([np.concatenate([atom, rule_head, args, np.array([score])])]))
        return embedding


# TODO: the prep_*_example methods should be generalized to use the EmbeddingMethod class
def prep_termwalk_example(example, vocab: Vocabulary) -> torch.Tensor:
    clean_atom(example[0])
    clean_rule(example[1])
    goal = termwalk.termwalk_representation(
        example[0], vocab)
    rule = termwalk.termwalk_representation(
        example[1], vocab)
    query = np.concatenate([goal, rule])
    score = example[2]
#    if score > 0:
#        print(score)
    embedding = torch.from_numpy(
        np.array([np.concatenate([query, np.array([score])])])).to_sparse()
    return embedding


def prep_chainbased_example(example, embed_size) -> torch.Tensor:
    clean_atom(example[0])
    clean_rule(example[1])
    goal = chainbased.represent_pattern(example[0], embed_size)
    rule = chainbased.represent_pattern(example[1], embed_size)
    query = np.concatenate([goal, rule])
    score = example[2]
#    if score > 0:
#        print(score)
    embedding = torch.from_numpy(
        np.array([np.concatenate([query, np.array([score])])]))
    return embedding


def prep_data(vocab: Vocabulary,
              a_path: str = "train_anchors.csv",
              p_path: str = "train_positives.csv",
              n_path: str = "train_negatives.csv",
              num_triplets: int = 70000):
    """ Prepares data for training the unification embedding model. Data consists of anchor, pos, and
    neg atoms, where the pos unifies with the anchor and the neg does not. The embedded versions of
    this data is written to three files. Optionally generates a new random KB.

    :param vocab: Data structure that defines the predicates, constants and variables used in a KB
    :param a_path:
    :param p_path:
    :param n_path:
    :param num_triplets:
    """
    # pred_list = [Predicate(3,'p0'),Predicate(1,'p1'),Predicate(1,'p2'),Predicate(2,'p3'),Predicate(2,'p4'),Predicate(2,'p5'),Predicate(1,'p6'),Predicate(4,'p7'), Predicate(2,'p8'),Predicate(2,'p9')]

    num_triplets = max(int(len(vocab.predicates) *
                       ((len(vocab.constants) + len(vocab.variables)) ** vocab.maxArity) / 1000), num_triplets)

    return atomgenerator.create_unity_embeddings(
        vocab, a_path, p_path, n_path, num_triplets)





# performs backward chaining; takes KB, path, and max depth and returns a list of all possible solutions


def backwardchain(KB: KnowledgeBase,
                  path_obj: Path,
                  max_depth, examples,
                  examples_list,
                  make_neg_facts: bool = True):

    global node_count
    node_count = 0
    ans_count = 0

    query = path_obj.node
    vars = set()
    for arg in query.arguments:
        if isinstance(arg, Variable):
            vars.add(copy(arg))
    vars = list(vars)
    G = Rule(Atom(Predicate(len(vars), "yes"), copy(vars)), [query])
    t = time.process_time()
    answers = backwardmain(KB, G, vars, path_obj, max_depth,
                           examples, examples_list, t, make_neg_facts)
    # since backwardchain uses yield, we have to loop through the results to get anything
    for _ in answers:
        ans_count = ans_count + 1
        # The following limit was necessary to make the process terminate on the query person(X)
        # for the LUBM unary data
        if ans_count >= MAX_TRAIN_ANSWERS:
            break
        # print('*',end='')
    if ans_count != 0:
        clear_line()
        print("Nodes: " + str(node_count))
    return ans_count


def backwardmain(KB: KnowledgeBase,
                 G: Rule, vars,
                 path_obj: Path,
                 max_depth: reasoner.MaxDepth,
                 examples,          # TODO: do we need this param? What is its purpose?
                 examples_list,
                 start_time,
                 make_neg_facts=True):
    '''
    Tries to prove goal by applying backward chaining until it reaches max depth or no new info is inferred.
    Records goal/rule training examples along the way.
    '''

    global node_count
    node_count = node_count + 1

    if node_count % 10000 == 1:
        diff = time.process_time() - start_time
        print_progress_bar(node_count, int(500000 * 1.5),
                           suffix=f'to max nodes ({int(node_count / diff) if diff > 0 else "-"} nps)', length=5, shown='percent')

    # Limit the search, by restricting the depth after a maximum number of nodes
    if node_count > 500000 and max_depth.max > 3:   # due to the 1.5 factor, limits to 5
        max_depth.max = 3

    if G.body:      # there are goals that must be proven

        # only check depth limit for a non-empty query
        if path_obj.depth > int(max_depth.max * 1.5):  # depth limiter
            path_obj.inc_fail()
            # example = (path_obj.root.node, path_obj.root_rule, path_obj.get_sf())
            # prep_regressor_example(example)
            # path_obj.push_root()
            return False

        # don't shuffle all the goals, only shuffle the order the body is added in, keep it depth-first
        # a1 = G.body.pop(random.choice(range(len(G.body))))
        a1 = G.body.pop(0)     # select first goal, and remove it from g's body

        path_obj.set_node(a1)

        no_ans = True
        # note, this will rearrange actual KB, should be OK
        random.shuffle(KB.rule_by_pred[a1.predicate])
        hasMatches = False
        for rule in KB.rule_by_pred[a1.predicate]:
            # standardizes rule
            rule_1 = copy(rule)

            reasoner.standardize(rule_1, path_obj.depth)

            subst = cache.unify_memoized(a1, rule_1.head)
            if isinstance(subst, dict):       # if rule's head unifies with the selected goal

                # check for cycles
                # experiments show that the process works better without this code
                # if len(rule.body) > 0:       # can't have a cycle when matching a fact
                #     node = path_obj
                #     cycle = False
                #     while node is not None:
                #         if node.parent is not None and rule == node.rule:
                #             # reuse of a rule when the goal is more specific: e.g. p(X,Y) vs. p(X,a)
                #             # That should be allowed?
                #             if isinstance(reasoner.unify(a1, node.parent.node), dict):    # goals unify
                #                 cycle = True
                #                 break
                #         node = node.parent
                #     if cycle:
                #         continue     # exit this iteration of for loop, try next rule

                hasMatches = True

                # this ensures that the order of the body does not impact reasoning
                random.shuffle(rule_1.body)
                new_body = rule_1.body + G.body

                # new_leaf = path_obj.get_leaf(rule,None)

                new_G = (Rule(reasoner.dosubst(G.head, subst), [
                         reasoner.dosubst(atom, subst) for atom in new_body]))
                new_leaf = path_obj.make_child(
                    a1, rule, new_body, len(rule_1.body))

                # TODO: I think this might be redundant now that the scoring has been fixed
                # solved the current goal a1 with a fact, base case for success
                if len(rule_1.body) == 0:
                    new_leaf.inc_success()

                for ret_val in backwardmain(KB, new_G, vars, new_leaf, max_depth, examples, examples_list, start_time):
                    yield ret_val
                    no_ans = False

        if no_ans:

            if path_obj.parent is not None:
                parent_goal_starts = path_obj.parent.goal_starts

                # create a negative example from a1 and the goal prior to substitutions
                # can only do this reliably when a fact was used in the previous step
                if (make_neg_facts and not hasMatches and len(parent_goal_starts) > 1
                        and len(path_obj.all_goals) < len(path_obj.parent.all_goals)):
                    # TODO: I don't think we need to sanitize here, given my fix...
                    # goal_pre_sub = vocab.sanitize_atom(path_obj.parent.all_goals[1])
                    # example = (goal_pre_sub, vocab.sanitize_atom(a1), 0)
                    goal_pre_sub = path_obj.parent.all_goals[1]
                    example = (goal_pre_sub, a1, 0)
                    if goal_pre_sub.predicate != a1.predicate:
                        print("Mismatched predicates: " + goal_pre_sub + " with " + str(a1))
                    examples_list.append(example)

            else:
                parent_goal_starts = []

            # TODO: need to thoroughly test this
            if path_obj.fail_level == -1:          # this is the end of the failed path
                # the fail level is the level the original failing goal was introduced at
                fail_level = path_obj.goal_starts[0]
            else:
                # the fail level was set by the successor node
                fail_level = path_obj.fail_level

            # any node whose target goal was introduced at an earlier level also fails
            # the failing nodes level
            if fail_level > path_obj.node_level:
                path_obj.inc_fail()
            else:
                # any parent node whose goal was introduced later (or at same level) had to have been successful to get here
                path_obj.inc_success()

            # if path_obj.parent is None:
            #     print("Failed to answer query at all...")
            # else:
            if path_obj.parent is not None:
                path_obj.parent.fail_level = fail_level   # pass the fail level up the tree
        else:
            path_obj.inc_success()          # once we succeed, we succeed all the way up the tree

        # we do this test because the root path_obj has no target or rule
        if (path_obj.parent != None):
            # example = (path_obj.parent.node, path_obj.rule, path_obj.get_sf())

            # note, there should not be a negative example if none of the rules matched
            # example = (path_obj.target, path_obj.rule, path_obj.get_sf())
            # TODO: I don't think we need to sanitize here, given my fix...
            example = (vocab.sanitize_atom(path_obj.target),
                       vocab.sanitize_rule(path_obj.rule), path_obj.get_score())
            examples_list.append(example)
#            prep_one_example(example, embeddings)

        if no_ans:
            return False

    else:  # no subgoals remain, we have proven the query successfully
        # print(f"depth = {path_obj.depth}")
        if path_obj.depth < max_depth.max and max_depth.max > 5:
            clear_line()
            print(f"min depth: {path_obj.depth}")
            # the following can significantly cutoff search, especially if an answer is found at depth=1
            # max_depth.set(path_obj.depth)   # this cuts off search for deeper answers than the one found
            max_depth.set(max(path_obj.depth, 5))

        path_obj.inc_success()
        # example = (path_obj.target, path_obj.rule, path_obj.get_sf())
        # TODO: I don't think we need to sanitize here, given my fix...
        example = (vocab.sanitize_atom(path_obj.target),
                   vocab.sanitize_rule(path_obj.rule), path_obj.get_score())
        examples_list.append(example)
#        prep_one_example(example, embeddings)
        # path_obj.push_root()

        if DEBUG:
            print("Solution: ")
            path_obj.print_rule_path()
            print()

        yield {vars[i]: G.head.arguments[i] for i in range(len(vars))}

# moved from rl-exp.py
# Generates KB of random queries


def generate_queries(facts: KnowledgeBase, num_queries: int, vocab: Vocabulary,
                     verbose: int = 1):
    queries = []
    for i in range(num_queries):
        if verbose >= 2:
            print(f"Query {i+1}/{num_queries}")
        query = Rule(reasoner.gen_random_query_vocab(facts.rules, vocab),
                     [])
        dupes = 0
        while query in queries:
            query = Rule(reasoner.gen_random_query_vocab(facts.rules, vocab),
                         [])
            dupes += 1
            if dupes > 10:
                break
        if dupes <= 10:
            queries.append(query)
    # print("custom query:")
    # customQuery = [Rule(Atom(Predicate(2, 'p0'), [Variable(
    #     'X1'), Constant('a2'), Constant('a2'), Constant('a2')]), [])]
    # reasoner.gen_random_query(customQuery)
    print("Number of queries: " + str(len(queries)))
    queries = KnowledgeBase(queries)
    return queries


# calls prep_examples(), which prepares dataset for training and testing the model by
# loading the dataset, tokenizing input text, and splitting it into training and testing sets
if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    # subparser = aparser.add_subparsers(title="subcommands",dest="subcommand")

    # KB options
    aparser.add_argument("--generate_kb", action="store_true",
                         help="Generate a random knowledge base while preparing the data.")
    aparser.add_argument("--num_rules", type=int, default=150)
    aparser.add_argument("--kb_path", default="randomKB.txt",
                         help="Location of the knowledge base text file. If --prep_data is on, this path will be used to save the knowledge base. Default: randomKB.txt")

    # Vocab options
    aparser.add_argument("--new_vocab", action="store_true",
                         help="Generates a new vocabulary, instead of reusing the one from randomKB.txt")
    aparser.add_argument("--vocab_file", default="vocab",
                         help="Path to dave generated vocab to.")
    aparser.add_argument("--vocab_from_kb", action="store_true",
                         help="If you want to create vocab from kb (path from --kb_vocab).")
    aparser.add_argument("--save_vocab", action="store_true",
                         help="If you want to save vocab to file (path from --vocab_file).")
    aparser.add_argument("--num_pred", type=int, default=10,
                         help="Number of predicates. Default: 10")
    aparser.add_argument("--num_const", type=int, default=100,
                         help="Number of constants. Default: 100")

    # Options for training the embedding model
    aparser.add_argument("-d", "--prep_data", action="store_true",
                         help="Prepare the data for training the unification embedding model")
    aparser.add_argument("--triplets", type=int, default=70000,
                         help="Number of triplets to generate to train the unity embedding model")
    aparser.add_argument("-e", "--embed_size", type=int, default=50,
                         help="Embed size. Defaults to 50")
    aparser.add_argument("--train_unification_model", action="store_true",
                         help="Train the model that produces the unification embeddings.")
    aparser.add_argument("--unification_model_path", default="rKB_model.pth",
                         help="Path to the unification representation model. If --train_unification_model is on, saves the model to this path. Default: rKB_model.pth.")
    aparser.add_argument("--train_autoencoder_model", action="store_true",
                         help="Train the model that produces the autoencoder embeddings. Argument should be the path at which the model is saved.")
    aparser.add_argument("--autoencoder_model_path", default="auto_encoder.pth",
                         help="Path to the autoencoder model. If --train_autoencoder_model is on, saves the model to this path. Default: auto_encoder.pth.")


    # TODO: remove these options once we've verified that the three triplet file are no longer used
    aparser.add_argument("--anchor_path", default="train_anchors.csv",
                         help="Location at which to save the anchor training data. Requires --prep_data. Default: train_anchors.csv.")
    aparser.add_argument("--positives_path", default="train_positives.csv",
                         help="Location at which to save the positive training data. Requires --prep_data. Default: train_positives.csv")
    aparser.add_argument("--negatives_path", default="train_negatives.csv",
                         help="Location at which to save the negative training data. Requires --prep_data. Default: train_negatives.csv")

    # Paths for saving the embeddings of training examples
    # We no longer use any of these
    # aparser.add_argument("-u", "--unification_embedding",
    #                      help="Location to save unification embeddings of training examples, if any")
    # aparser.add_argument("-a", "--autoencoder_embedding",
    #                      help="Location to save autoencoder embeddings of training examples, if any")
    # aparser.add_argument("-t", "--termwalk_embedding",
    #                      help="Location to save termwalk embeddings of training examples, if any")
    # aparser.add_argument("-c", "--chainbased_embedding",
    #                      help="Location to save chainbased embeddings of training examples, if any")

    # Options for training the scoring model
    aparser.add_argument("-p", "--prep_examples", action="store_true",
                         help="Generate (goal,rule) examples for training the scoring model.")
    aparser.add_argument("--no_neg_facts", action="store_true",
                         help="Turn off generating negative examples for facts.")
    aparser.add_argument("--train_example_path", default="mr_train_examples.csv", type=str,
                         help="Location to save the (goal,rule) training examples. Default: mr_train_examples.csv")

    aparser.add_argument("--save_unity_embeddings", default=False, action="store_true",
                         help="Save the unity embeddings of the training examples. Default: False")


    # Query generation
    aparser.add_argument("-g", "--generate_queries", action="store_true",
                         help="Generate new queries instead of reading them in.")
    aparser.add_argument("--facts_file", default="all_facts.txt",
                         help="Path to the list of facts. Default: all_facts.txt")
    aparser.add_argument("--train_query_path", default="train_queries.txt",
                         help="Path to save the list of queries used for \
                        training. Default: train_queries.txt")
    aparser.add_argument("--test_query_path", default="test_queries.txt",
                         help="Path to save the list of queries used to test \
                        the model. Default: test_queries.txt")
    aparser.add_argument("--num_queries", type=int, default=200,
                         help="Number of queries (training + test). Default: 200")

    # not currently using this functionality
    # aparser.add_argument("-m", "--map_data", action="store_true",
    #                      help="Translate the original knowledgebase previously generated into a generic semantic set")
    # # aparser.add_argument('integers',type=int,nargs='3',help='an integer collection set')
    # aparser.add_argument('--filename', type=str,
    #                      help='Name of file you wish to access to generate semantics set from for map_data or map_data_from_semantic_set methods')
    # aparser.add_argument('--list_of_pred_arities', type=int, nargs='+',
    #                      help='list of arity levels to generate semantic set mappings for map_data_from_symbol_set method')
    # aparser.add_argument("-n", "--map_data_from_symbol_set", action="store_true",
    #                      help="Translate the original knowledgebase previously generic into a generic semantic set from a specified symbol set")
    # aparser.add_argument("-f", "--organize_symbol_set", action="store_true",
    #                      help="Just here to test one hot encoder method for semantic set arities")

    args = aparser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If user doesn't request generation of a new vocabulary, then reuse the predicates from the latest KB
    vocab = Vocabulary()
    if args.new_vocab:
        vocab.random_init(num_pred=args.num_pred,
                          num_const=args.num_const, arity_dist=[0.5, 0.5, 0, 0, 0])
        num_classes = len(vocab.predicates) * \
            (len(vocab.constants) ** vocab.maxArity)
        print(f"Classes: {num_classes}")
        print(f"Max arity: {vocab.maxArity}")

        # print("KB embed size: " + str(get_embed_size(vocab)))
        # save vocab by default!
        #if args.save_vocab:
        vocab.save_vocab_to_file(args.vocab_file)
    else:
        if args.vocab_from_kb:
            vocabKB = kbparser.parse_KB_file(args.kb_path)
            vocab.init_from_kb(vocabKB)
            print("Creating vocabulary from last " +
                  args.kb_path + " knowledge base")
            # save vocab by default!
            # if args.save_vocab:
            vocab.save_vocab_to_file(args.vocab_file)
        else:
            vocab.init_from_vocab(args.vocab_file)

    if args.prep_data:
        prep_data(vocab, args.anchor_path, args.positives_path,
                  args.negatives_path, args.triplets)

    if args.train_unification_model:
        nnunifier.generate_unification_model(
            args.anchor_path, args.positives_path, args.negatives_path, args.unification_model_path, vocab, args.embed_size, args.save_unity_embeddings)
        print("unification model generated")

    if args.train_autoencoder_model:
        autoencoder.generate_auto_model(
            args.anchor_path, args.positives_path, args.negatives_path, args.autoencoder_model_path)
        print("autoencoder model generated")

    # JDH: moved out of prep_data, because it is a different feature that we may want to run independently
    if args.generate_kb:
        KB = generate_random_KB(vocab, args.num_rules)
        kbparser.KB_to_txt(KB, args.kb_path)
        print(f"KB generated ({len(KB.rules)})")

    if args.prep_examples:
        make_neg_facts = not args.no_neg_facts
        prep_examples(args.train_example_path, args.kb_path,
                      args.train_query_path, make_neg_facts)

    # NOT USING THIS FUNCTIONALITY
    # if args.map_data:
    #     if args.filename:
    #         filename = args.filename
    #         map_data(filename)
    #     else:
    #         raise Exception(
    #             "ERROR: Must include filename after execution of -m flag [Command written as python kbencoder.py -m --filename filename]")
    #
    # if args.map_data_from_symbol_set:
    #     if args.filename and args.list_of_pred_arities:
    #         list_of_pred_arities = args.list_of_pred_arities
    #         filename = args.filename
    #         map_data_from_symbol_set(
    #             filename, len(vocab.constants), len(vocab.variables), list_of_pred_arities)
    #     else:
    #         raise Exception("ERROR: Must include number of constants, number of variables, and list of predicate arities after "
    #                         + "execution of -n flag [Command written as python kbencoder.py -n --filename filename --list_of_pred_arities list_of_pred_arities]")

    # First: generate the training and testing datasets.
    # Second: train the model.
    # Third: Run the experiment with both the standard and guided reasoner.
    # Perhaps also with the alternative choice strategy?
    if args.generate_queries:
        print("Generating facts list...")
        KB = kbparser.parse_KB_file(args.kb_path)
        facts, max_depth = gen_all_facts(KB, vocab)
        facts_kb = KnowledgeBase(facts)
        kbparser.KB_to_txt(facts_kb, args.facts_file)
        print("Facts list generated")
        print(f"{len(facts)} total facts. Max depth = {max_depth}")
        print()

        # facts = kbparser.parse_KB_file(args.facts_file)
        queries = generate_queries(facts_kb, args.num_queries, vocab)
        train_queries = KnowledgeBase(queries.rules[::2])
        kbparser.KB_to_txt(train_queries, args.train_query_path)
        print(f"{len(train_queries)} training queries generated")

        test_queries = KnowledgeBase(queries.rules[1::2])
        kbparser.KB_to_txt(test_queries, args.test_query_path)
        print(f"{len(test_queries)} testing queries generated")
